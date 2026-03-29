"""
Whisper GUI (Tkinter) with Login & Register using MySQL (XAMPP) - Enhanced Design
---------------------------------------------------------------------------------
Single-file app:
- Login / Register pages (credentials stored in MySQL via XAMPP)
- After login: GUI for Whisper transcription with options
- Speaker detection (voice-based if librosa+sklearn available; else basic)
- Output saved to auto-named .txt in chosen folder + shown in app

Prerequisites:
1) Python 3.10+
2) pip install:
   pip install openai-whisper
   pip install mysql-connector-python
   pip install numpy scipy scikit-learn librosa
   pip install soundfile
   
   (Optional voice analysis will use librosa+sklearn; if missing, app falls back.)

3) FFmpeg installed & on PATH (required by Whisper)
   - Windows: https://ffmpeg.org → add /bin to PATH

4) Start XAMPP MySQL and set credentials below (default root/no password).

Database bootstrap (runs automatically on first start):
- Creates DB `whisper_app` and table `users` if not present.

Run:
   python whisper_gui_mysql.py
"""

import os
import threading
import time
from datetime import datetime
import hashlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# -------------------- Optional voice analysis deps --------------------
VOICE_ANALYSIS_AVAILABLE = False
try:
    import numpy as np
    import librosa
    from sklearn.cluster import KMeans
    VOICE_ANALYSIS_AVAILABLE = True
except Exception:
    VOICE_ANALYSIS_AVAILABLE = False

# -------------------- Whisper import (with friendly error) ------------
WHISPER_AVAILABLE = True
try:
    import whisper
except Exception:
    WHISPER_AVAILABLE = False

# -------------------- MySQL connector (XAMPP) -------------------------
DB_AVAILABLE = True
try:
    import mysql.connector as mysql
except Exception:
    DB_AVAILABLE = False

# -------------------- Configuration ----------------------------------
DB_HOST = "localhost"
DB_USER = "root"       # XAMPP default
DB_PASS = ""           # XAMPP default empty
DB_NAME = "whisper_app"
USERS_TABLE = "users"

# -------------------- Color Scheme & Styling -------------------------
COLORS = {
    'primary': '#2E3440',      # Dark blue-gray
    'secondary': '#3B4252',    # Lighter blue-gray
    'accent': '#5E81AC',       # Blue accent
    'success': '#A3BE8C',      # Green
    'warning': '#EBCB8B',      # Yellow
    'error': '#BF616A',        # Red
    'text': '#2E3440',         # Dark text
    'light_text': '#4C566A',   # Light gray text
    'bg': '#ECEFF4',           # Light background
    'white': '#FFFFFF',        # White
    'border': '#D8DEE9'        # Light border
}

# -------------------- Utility funcs (from user's script) --------------
def generate_output_filename(input_file, output_folder):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    filename = f"{base_name}.txt"
    return os.path.join(output_folder, filename)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def extract_voice_features(audio_file, segments):
    if not VOICE_ANALYSIS_AVAILABLE:
        return None
    try:
        y, sr = librosa.load(audio_file, sr=16000)
        voice_features = []
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = y[start_sample:end_sample]
            if len(segment_audio) < sr * 0.5:
                voice_features.append(None)
                continue
            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            pitches, _ = librosa.piptrack(y=segment_audio, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            features = np.concatenate([mfcc_mean, mfcc_std, [pitch_mean]])
            voice_features.append(features)
        return voice_features
    except Exception:
        return None

def cluster_speakers_by_voice(voice_features, max_speakers=4):
    if voice_features is None or not VOICE_ANALYSIS_AVAILABLE:
        return None
    valid_features, valid_indices = [], []
    for i, f in enumerate(voice_features):
        if f is not None:
            valid_features.append(f)
            valid_indices.append(i)
    if len(valid_features) < 2:
        return None
    try:
        features_array = np.array(valid_features)
        n_speakers = min(max_speakers, max(2, len(valid_features) // 10))
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_array)
        speaker_assignments = [0] * len(voice_features)
        for i, idx in enumerate(valid_indices):
            speaker_assignments[idx] = labels[i] + 1
        return speaker_assignments
    except Exception:
        return None

def detect_speaker_changes(segments, audio_file=None, use_voice_analysis=True):
    if not segments:
        return segments
    speakers_text = []
    if use_voice_analysis and VOICE_ANALYSIS_AVAILABLE and audio_file:
        voice_features = extract_voice_features(audio_file, segments)
        speaker_assignments = cluster_speakers_by_voice(voice_features)
        if speaker_assignments:
            for i, seg in enumerate(segments):
                spk_num = speaker_assignments[i] if i < len(speaker_assignments) else 1
                speakers_text.append({
                    'start': seg['start'], 'end': seg['end'], 'text': seg['text'].strip(), 'speaker': f"Speaker {spk_num}"
                })
            return speakers_text
    # fallback basic
    current_speaker = 1
    max_speakers = 4
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        change = False
        if i > 0:
            prev_end = segments[i-1]['end']
            pause = seg['start'] - prev_end
            if pause > 4.0:
                change = True
            clear_markers = [
                'hello','hi there','yes sir','no sir','thank you very much','excuse me','i think','well then','okay so',
                'हैलो','नमस्ते','हां जी हां','नहीं जी','धन्यवाद','माफ करिए'
            ]
            tl = text.lower()
            if any(tl.startswith(m) for m in clear_markers) and pause > 2.0:
                change = True
        if change:
            current_speaker = (current_speaker % max_speakers) + 1
        speakers_text.append({
            'start': seg['start'], 'end': seg['end'], 'text': text, 'speaker': f"Speaker {current_speaker}"
        })
    return speakers_text

def format_transcript_with_speakers(segments_with_speakers):
    out = []
    for seg in segments_with_speakers:
        ts = format_time(seg['start'])
        out.append(f"[{ts}] {seg['speaker']}: {seg['text']}")
    return "\n\n".join(out)

# -------------------- Database Layer ----------------------------------
class DB:
    def __init__(self, host, user, password):
        self.host = host
        self.user = user
        self.password = password
        self._ensure_db()

    def _conn(self, database=None):
        if not DB_AVAILABLE:
            raise RuntimeError("mysql-connector-python not installed")
        kwargs = dict(host=self.host, user=self.user, password=self.password)
        if database:
            kwargs["database"] = database
        return mysql.connect(**kwargs)

    def _ensure_db(self):
        if not DB_AVAILABLE:
            return
        # create DB & table if not exists
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.commit()
        cur.close()
        conn.close()
        conn = self._conn(DB_NAME)
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {USERS_TABLE} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(128) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
        """)
        conn.commit()
        cur.close()
        conn.close()

    @staticmethod
    def _hash(pw: str) -> str:
        return hashlib.sha256(pw.encode('utf-8')).hexdigest()

    def register(self, username: str, password: str) -> bool:
        conn = self._conn(DB_NAME)
        try:
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO {USERS_TABLE} (username, password) VALUES (%s, %s)",
                (username, self._hash(password))
            )
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close(); conn.close()

    def validate(self, username: str, password: str) -> bool:
        conn = self._conn(DB_NAME)
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT password FROM {USERS_TABLE} WHERE username=%s",
                (username,)
            )
            row = cur.fetchone()
            if not row:
                return False
            return row[0] == self._hash(password)
        finally:
            cur.close(); conn.close()

# -------------------- Custom Widgets ----------------------------------
class CustomButton(ttk.Button):
    """Enhanced button with better styling"""
    def __init__(self, parent, text, command=None, style="Accent.TButton", **kwargs):
        super().__init__(parent, text=text, command=command, style=style, **kwargs)

class CustomEntry(ttk.Entry):
    """Enhanced entry with placeholder support"""
    def __init__(self, parent, placeholder="", **kwargs):
        super().__init__(parent, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = COLORS['light_text']
        self.normal_color = COLORS['text']
        
        if placeholder:
            self.insert(0, placeholder)
            self.config(foreground=self.placeholder_color)
            self.bind('<FocusIn>', self._on_focus_in)
            self.bind('<FocusOut>', self._on_focus_out)
    
    def _on_focus_in(self, event):
        if self.get() == self.placeholder:
            self.delete(0, tk.END)
            self.config(foreground=self.normal_color)
    
    def _on_focus_out(self, event):
        if not self.get():
            self.insert(0, self.placeholder)
            self.config(foreground=self.placeholder_color)
    
    def get_value(self):
        value = self.get()
        return "" if value == self.placeholder else value

# -------------------- GUI Frames --------------------------------------
class LoginFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.build()

    def build(self):
        # Configure main container
        self.configure(style='Card.TFrame')
        
        # Create main container with gradient effect
        main_container = ttk.Frame(self, style='Main.TFrame')
        main_container.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Login card
        login_card = ttk.Frame(main_container, style='Card.TFrame')
        login_card.pack(expand=True, fill='both', padx=60, pady=60)
        login_card.columnconfigure(0, weight=1)
        
        # Header section
        header_frame = ttk.Frame(login_card, style='Header.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 40))
        header_frame.columnconfigure(0, weight=1)
        
        # App icon/logo (using text for now)
        logo_frame = ttk.Frame(header_frame)
        logo_frame.grid(row=0, column=0, pady=(0, 20))
        
        logo_label = ttk.Label(logo_frame, text="🎤", font=("Segoe UI", 48))
        logo_label.pack()
        
        title = ttk.Label(header_frame, text="Whisper Transcription", 
                         font=("Segoe UI", 24, "bold"), style='Title.TLabel')
        title.grid(row=1, column=0)
        
        subtitle = ttk.Label(header_frame, text="Advanced Speech-to-Text Processing", 
                           font=("Segoe UI", 12), style='Subtitle.TLabel')
        subtitle.grid(row=2, column=0, pady=(5, 0))

        # Form section
        form_frame = ttk.Frame(login_card, style='Form.TFrame')
        form_frame.grid(row=1, column=0, sticky="ew", pady=(0, 30))
        form_frame.columnconfigure(0, weight=1)

        # Username field
        user_container = ttk.Frame(form_frame)
        user_container.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        user_container.columnconfigure(0, weight=1)
        
        ttk.Label(user_container, text="Username", 
                 font=("Segoe UI", 11, "bold"), style='FieldLabel.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 8))
        
        self.username = CustomEntry(user_container, placeholder="Enter your username", 
                                   font=("Segoe UI", 11), style='Custom.TEntry')
        self.username.grid(row=1, column=0, sticky="ew", ipady=12)

        # Password field
        pass_container = ttk.Frame(form_frame)
        pass_container.grid(row=1, column=0, sticky="ew", pady=(0, 30))
        pass_container.columnconfigure(0, weight=1)
        
        ttk.Label(pass_container, text="Password", 
                 font=("Segoe UI", 11, "bold"), style='FieldLabel.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 8))
        
        self.password = ttk.Entry(pass_container, show="*", font=("Segoe UI", 11), style='Custom.TEntry')
        self.password.grid(row=1, column=0, sticky="ew", ipady=12)

        # Buttons section
        buttons_frame = ttk.Frame(login_card)
        buttons_frame.grid(row=2, column=0, sticky="ew")
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)

        login_btn = CustomButton(buttons_frame, text="Sign In", command=self.do_login, style="Primary.TButton")
        login_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=12)

        register_btn = CustomButton(buttons_frame, text="Create Account", command=self.app.show_register, style="Secondary.TButton")
        register_btn.grid(row=0, column=1, sticky="ew", padx=(10, 0), ipady=12)
        
        # Bind Enter key
        self.bind_all('<Return>', lambda e: self.do_login())

    def do_login(self):
        u = self.username.get_value() if hasattr(self.username, 'get_value') else self.username.get().strip()
        p = self.password.get().strip()
        if not u or not p:
            messagebox.showwarning("Missing Information", "Please enter both username and password")
            return
        try:
            if self.app.db.validate(u, p):
                self.app.current_user = u
                self.app.show_main()
            else:
                messagebox.showerror("Authentication Failed", "Invalid username or password")
        except Exception as e:
            messagebox.showerror("Database Error", str(e))

class RegisterFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.build()

    def build(self):
        # Configure main container
        self.configure(style='Card.TFrame')
        
        # Create main container
        main_container = ttk.Frame(self, style='Main.TFrame')
        main_container.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Register card
        register_card = ttk.Frame(main_container, style='Card.TFrame')
        register_card.pack(expand=True, fill='both', padx=60, pady=60)
        register_card.columnconfigure(0, weight=1)
        
        # Header section
        header_frame = ttk.Frame(register_card, style='Header.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 40))
        header_frame.columnconfigure(0, weight=1)
        
        # Back button
        back_frame = ttk.Frame(header_frame)
        back_frame.grid(row=0, column=0, sticky="w", pady=(0, 20))
        
        back_btn = CustomButton(back_frame, text="← Back to Login", command=self.app.show_login, style="Link.TButton")
        back_btn.pack(side='left')
        
        title = ttk.Label(header_frame, text="Create Account", 
                         font=("Segoe UI", 24, "bold"), style='Title.TLabel')
        title.grid(row=1, column=0)
        
        subtitle = ttk.Label(header_frame, text="Join the Whisper community today", 
                           font=("Segoe UI", 12), style='Subtitle.TLabel')
        subtitle.grid(row=2, column=0, pady=(5, 0))

        # Form section
        form_frame = ttk.Frame(register_card, style='Form.TFrame')
        form_frame.grid(row=1, column=0, sticky="ew", pady=(0, 30))
        form_frame.columnconfigure(0, weight=1)

        # Username field
        user_container = ttk.Frame(form_frame)
        user_container.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        user_container.columnconfigure(0, weight=1)
        
        ttk.Label(user_container, text="Username", 
                 font=("Segoe UI", 11, "bold"), style='FieldLabel.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 8))
        
        self.username = CustomEntry(user_container, placeholder="Choose a username", 
                                   font=("Segoe UI", 11), style='Custom.TEntry')
        self.username.grid(row=1, column=0, sticky="ew", ipady=12)

        # Password field
        pass_container = ttk.Frame(form_frame)
        pass_container.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        pass_container.columnconfigure(0, weight=1)
        
        ttk.Label(pass_container, text="Password", 
                 font=("Segoe UI", 11, "bold"), style='FieldLabel.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 8))
        
        self.password = ttk.Entry(pass_container, show="*", font=("Segoe UI", 11), style='Custom.TEntry')
        self.password.grid(row=1, column=0, sticky="ew", ipady=12)

        # Confirm Password field
        confirm_container = ttk.Frame(form_frame)
        confirm_container.grid(row=2, column=0, sticky="ew", pady=(0, 30))
        confirm_container.columnconfigure(0, weight=1)
        
        ttk.Label(confirm_container, text="Confirm Password", 
                 font=("Segoe UI", 11, "bold"), style='FieldLabel.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 8))
        
        self.password2 = ttk.Entry(confirm_container, show="*", font=("Segoe UI", 11), style='Custom.TEntry')
        self.password2.grid(row=1, column=0, sticky="ew", ipady=12)

        # Buttons section
        buttons_frame = ttk.Frame(register_card)
        buttons_frame.grid(row=2, column=0, sticky="ew")
        buttons_frame.columnconfigure(0, weight=1)

        create_btn = CustomButton(buttons_frame, text="Create Account", command=self.do_register, style="Primary.TButton")
        create_btn.grid(row=0, column=0, sticky="ew", ipady=12)
        
        # Bind Enter key
        self.bind_all('<Return>', lambda e: self.do_register())

    def do_register(self):
        u = self.username.get_value() if hasattr(self.username, 'get_value') else self.username.get().strip()
        p = self.password.get().strip()
        p2 = self.password2.get().strip()
        if not u or not p:
            messagebox.showwarning("Missing Information", "Please enter username and password")
            return
        if p != p2:
            messagebox.showwarning("Password Mismatch", "Passwords do not match")
            return
        try:
            self.app.db.register(u, p)
            messagebox.showinfo("Success", "Account created successfully! Please login.")
            self.app.show_login()
        except Exception as e:
            messagebox.showerror("Registration Error", str(e))

class MainFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_var = tk.StringVar(value="tiny")
        self.lang_var = tk.StringVar(value="auto")
        self.spk_var = tk.StringVar(value="voice")
        self.status_var = tk.StringVar(value="Ready to transcribe")
        self.progress_var = tk.IntVar()
        self.build()

    def build(self):
        self.configure(style='Main.TFrame')
        
        # Header with user info and logout
        header = ttk.Frame(self, style='Header.TFrame')
        header.pack(fill='x', padx=20, pady=(20, 0))
        header.columnconfigure(1, weight=1)
        
        # Logo and title
        logo_title_frame = ttk.Frame(header)
        logo_title_frame.grid(row=0, column=0, sticky='w')
        
        ttk.Label(logo_title_frame, text="🎤", font=("Segoe UI", 20)).pack(side='left', padx=(0, 10))
        ttk.Label(logo_title_frame, text="Whisper Transcription", 
                 font=("Segoe UI", 16, "bold"), style='Title.TLabel').pack(side='left')
        
        # User info and logout
        user_frame = ttk.Frame(header)
        user_frame.grid(row=0, column=2, sticky='e')
        
        ttk.Label(user_frame, text=f"Welcome, {self.app.current_user or 'User'}", 
                 font=("Segoe UI", 11), style='Subtitle.TLabel').pack(side='left', padx=(0, 15))
        
        logout_btn = CustomButton(user_frame, text="Logout", command=self.app.logout, style="Link.TButton")
        logout_btn.pack(side='left')

        # Main content area
        content = ttk.Frame(self, style='Content.TFrame')
        content.pack(fill='both', expand=True, padx=20, pady=20)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        # Left panel - Controls
        left_panel = ttk.Frame(content, style='Panel.TFrame')
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # File Selection Section
        file_section = ttk.LabelFrame(left_panel, text="📁 File Selection", 
                                     style='Section.TLabelframe', padding=20)
        file_section.pack(fill='x', pady=(0, 20))
        file_section.columnconfigure(0, weight=1)

        # Audio file selection
        ttk.Label(file_section, text="Audio File", 
                 font=("Segoe UI", 10, "bold"), style='FieldLabel.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 8))
        
        audio_frame = ttk.Frame(file_section)
        audio_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        audio_frame.columnconfigure(0, weight=1)
        
        self.audio_entry = ttk.Entry(audio_frame, textvariable=self.input_path, 
                                    font=("Segoe UI", 9), style='Path.TEntry', state='readonly')
        self.audio_entry.grid(row=0, column=0, sticky='ew', padx=(0, 10))
        
        audio_btn = CustomButton(audio_frame, text="Browse", command=self.pick_input, style="Secondary.TButton")
        audio_btn.grid(row=0, column=1)

        # Output folder selection
        ttk.Label(file_section, text="Output Folder", 
                 font=("Segoe UI", 10, "bold"), style='FieldLabel.TLabel').grid(row=2, column=0, sticky='w', pady=(0, 8))
        
        output_frame = ttk.Frame(file_section)
        output_frame.grid(row=3, column=0, sticky='ew')
        output_frame.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, 
                                     font=("Segoe UI", 9), style='Path.TEntry', state='readonly')
        self.output_entry.grid(row=0, column=0, sticky='ew', padx=(0, 10))
        
        output_btn = CustomButton(output_frame, text="Choose", command=self.pick_output_dir, style="Secondary.TButton")
        output_btn.grid(row=0, column=1)

        # Configuration Section
        config_section = ttk.LabelFrame(left_panel, text="⚙️ Configuration", 
                                       style='Section.TLabelframe', padding=20)
        config_section.pack(fill='x', pady=(0, 20))
        config_section.columnconfigure(1, weight=1)

        # Model selection
        ttk.Label(config_section, text="AI Model", 
                 font=("Segoe UI", 10, "bold"), style='FieldLabel.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 8))
        
        model_combo = ttk.Combobox(config_section, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large"], 
                                  state="readonly", font=("Segoe UI", 9), style='Custom.TCombobox')
        model_combo.grid(row=0, column=1, sticky='ew', pady=(0, 15))

        # Language selection
        ttk.Label(config_section, text="Language", 
                 font=("Segoe UI", 10, "bold"), style='FieldLabel.TLabel').grid(row=1, column=0, sticky='w', pady=(0, 8))
        
        lang_entry = ttk.Entry(config_section, textvariable=self.lang_var, 
                              font=("Segoe UI", 9), style='Custom.TEntry')
        lang_entry.grid(row=1, column=1, sticky='ew', pady=(0, 15))

        # Speaker detection
        ttk.Label(config_section, text="Speaker Detection", 
                 font=("Segoe UI", 10, "bold"), style='FieldLabel.TLabel').grid(row=2, column=0, sticky='w', columnspan=2, pady=(5, 8))
        
        speaker_frame = ttk.Frame(config_section)
        speaker_frame.grid(row=3, column=0, columnspan=2, sticky='ew')
        
        ttk.Radiobutton(speaker_frame, text="🎵 Voice-based", value="voice", 
                       variable=self.spk_var, style='Custom.TRadiobutton').pack(side='left', padx=(0, 15))
        ttk.Radiobutton(speaker_frame, text="📝 Basic", value="basic", 
                       variable=self.spk_var, style='Custom.TRadiobutton').pack(side='left', padx=(0, 15))
        ttk.Radiobutton(speaker_frame, text="❌ Off", value="off", 
                       variable=self.spk_var, style='Custom.TRadiobutton').pack(side='left')

        # Action Section
        action_section = ttk.Frame(left_panel, style='Action.TFrame')
        action_section.pack(fill='x', pady=(0, 20))
        
        # Main transcribe button
        self.transcribe_btn = CustomButton(action_section, text="🎯 Start Transcription", 
                                          command=self.start_transcription, style="Primary.TButton")
        self.transcribe_btn.pack(fill='x', ipady=15)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(action_section, mode='indeterminate', style='Custom.Horizontal.TProgressbar')
        self.progress_bar.pack(fill='x', pady=(15, 10))
        
        # Status label
        self.status_label = ttk.Label(action_section, textvariable=self.status_var, 
                                     font=("Segoe UI", 9), style='Status.TLabel', anchor='center')
        self.status_label.pack(fill='x')

        # Right panel - Output
        right_panel = ttk.Frame(content, style='Panel.TFrame')
        right_panel.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # Output header
        output_header = ttk.Frame(right_panel)
        output_header.grid(row=0, column=0, sticky='ew', pady=(0, 15))
        output_header.columnconfigure(0, weight=1)
        
        ttk.Label(output_header, text="📄 Transcript Preview", 
                 font=("Segoe UI", 14, "bold"), style='SectionTitle.TLabel').grid(row=0, column=0, sticky='w')
        
        # Clear button
        clear_btn = CustomButton(output_header, text="Clear", command=self.clear_output, style="Link.TButton")
        clear_btn.grid(row=0, column=1, sticky='e')
        
        # Output text area with custom styling
        output_frame = ttk.Frame(right_panel, style='Output.TFrame')
        output_frame.grid(row=1, column=0, sticky='nsew')
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        self.preview = ScrolledText(output_frame, wrap='word', font=("Consolas", 10), 
                                   bg=COLORS['white'], fg=COLORS['text'], 
                                   selectbackground=COLORS['accent'], selectforeground='white',
                                   relief='flat', borderwidth=0, padx=15, pady=15)
        self.preview.grid(row=0, column=0, sticky='nsew')
        
        # Configure text tags for better formatting
        self.preview.tag_configure("header", font=("Segoe UI", 11, "bold"), foreground=COLORS['primary'])
        self.preview.tag_configure("timestamp", font=("Consolas", 9), foreground=COLORS['accent'])
        self.preview.tag_configure("speaker", font=("Segoe UI", 10, "bold"), foreground=COLORS['success'])
        self.preview.tag_configure("content", font=("Segoe UI", 10), foreground=COLORS['text'])

    def pick_input(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.input_path.set(path)

    def pick_output_dir(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_dir.set(d)
    
    def clear_output(self):
        self.preview.delete("1.0", tk.END)

    def start_transcription(self):
        if not WHISPER_AVAILABLE:
            messagebox.showerror("Missing Dependency", 
                               "OpenAI Whisper is not installed.\n\nPlease run: pip install openai-whisper")
            return
        
        audio = self.input_path.get().strip()
        outdir = self.output_dir.get().strip()
        
        if not audio or not os.path.isfile(audio):
            messagebox.showwarning("Audio File Missing", "Please select a valid audio file")
            return
        if not outdir:
            messagebox.showwarning("Output Folder Missing", "Please choose an output folder")
            return
        
        os.makedirs(outdir, exist_ok=True)
        self.clear_output()
        
        # Start progress animation
        self.progress_bar.start(10)
        self.transcribe_btn.config(state='disabled', text='🔄 Processing...')
        self.status_var.set("Initializing transcription...")
        
        # Start transcription in separate thread
        thread = threading.Thread(target=self._transcribe_thread, args=(audio, outdir), daemon=True)
        thread.start()

    def _transcribe_thread(self, audio_path, outdir):
        try:
            model_name = self.model_var.get()
            language = self.lang_var.get().strip().lower()
            if language in ("auto", ""):
                language = None
            use_speakers = self.spk_var.get() != "off"
            use_voice = self.spk_var.get() == "voice"

            self._set_status("Loading AI model...")
            model = whisper.load_model(model_name)

            self._set_status("Processing audio... This may take several minutes")
            start = time.time()
            result = model.transcribe(
                audio_path,
                language=language,
                fp16=False,
                verbose=False,
                word_timestamps=True if use_speakers else False,
                temperature=0.0,
            )
            elapsed = time.time() - start
            text = result.get("text", "").strip()

            speaker_transcript = ""
            if use_speakers and 'segments' in result:
                self._set_status("Analyzing speakers and formatting...")
                segs = detect_speaker_changes(result['segments'], audio_file=audio_path if use_voice else None, use_voice_analysis=use_voice)
                speaker_transcript = format_transcript_with_speakers(segs)

            output_file = generate_output_filename(audio_path, outdir)

            report = []
            report.append("ENHANCED AUDIO TRANSCRIPTION REPORT")
            report.append("="*60)
            report.append("👨‍💻 Developed by: Aditi Pandit")
            report.append("="*60)
            report.append(f"📁 Original File: {os.path.basename(audio_path)}")
            report.append(f"📅 Transcription Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"⏱️  Processing Time: {format_time(elapsed)}")
            report.append(f"🌍 Detected Language: {result.get('language','Unknown').upper()}")
            report.append(f"🤖 Model Used: {model_name.upper()}")
            report.append(f"📊 Text Length: {len(text)} characters")
            report.append(f"📝 Estimated Words: {len(text.split())} words")
            report.append(f"🎤 Speaker Detection: {'Enabled' if use_speakers else 'Disabled'}\n")
            report.append("="*60)
            report.append("FULL TRANSCRIPT (Plain Text):")
            report.append("="*60)
            report.append("")
            report.append(text)
            report.append("")
            if use_speakers and speaker_transcript:
                report.append("="*60)
                report.append("SPEAKER-SEPARATED TRANSCRIPT:")
                report.append("="*60)
                report.append("")
                report.append(speaker_transcript)
                report.append("")
            report.append("="*60)
            report.append("END OF TRANSCRIPT")
            report.append("="*60)
            report.append("")
            report.append("🎵 Generated by Enhanced Whisper Speech-to-Text")
            report.append("👨‍💻 Developed by Aditi Pandit")
            report.append(f"📁 Filename: {os.path.basename(output_file)}")
            report.append("⚡ Features: Auto-transcription, Speaker detection, Name recognition\n")

            final_text = "\n".join(report)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_text)

            # Update preview in UI thread with formatting
            self.preview.after(0, lambda: self._update_preview(final_text))
            self._set_status(f"✅ Transcription completed! Saved to: {os.path.basename(output_file)}")
            
            # Show success message
            self.preview.after(0, lambda: messagebox.showinfo("Success", 
                f"Transcription completed successfully!\n\nSaved to:\n{output_file}\n\nProcessing time: {format_time(elapsed)}"))
            
        except Exception as e:
            self._set_status("❌ Transcription failed")
            self.preview.after(0, lambda: messagebox.showerror("Transcription Error", str(e)))
        finally:
            # Reset UI state
            self.preview.after(0, self._reset_ui)

    def _update_preview(self, text):
        """Update preview with formatted text"""
        self.preview.delete("1.0", tk.END)
        lines = text.split('\n')
        
        for line in lines:
            if line.startswith('='):
                self.preview.insert(tk.END, line + '\n', "header")
            elif line.startswith('[') and ']' in line:
                # Timestamp and speaker formatting
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    timestamp_speaker = parts[0] + ': '
                    content = parts[1]
                    self.preview.insert(tk.END, timestamp_speaker, "speaker")
                    self.preview.insert(tk.END, content + '\n', "content")
                else:
                    self.preview.insert(tk.END, line + '\n', "content")
            else:
                self.preview.insert(tk.END, line + '\n', "content")
        
        self.preview.see("1.0")  # Scroll to top

    def _reset_ui(self):
        """Reset UI to ready state"""
        self.progress_bar.stop()
        self.transcribe_btn.config(state='normal', text='🎯 Start Transcription')

    def _set_status(self, txt):
        self.status_var.set(txt)

# -------------------- Root App with Enhanced Styling ------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper AI - Advanced Speech Transcription")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Set window icon (using emoji for now)
        try:
            # You can replace this with an actual .ico file
            self.iconbitmap(default='')  # Add path to .ico file if available
        except:
            pass
        
        self._setup_styles()
        
        if not DB_AVAILABLE:
            messagebox.showwarning("Database Warning", 
                                 "MySQL connector not available.\n\nPlease install: pip install mysql-connector-python")
        
        self.db = DB(DB_HOST, DB_USER, DB_PASS) if DB_AVAILABLE else None
        self.current_user = None

        self.container = ttk.Frame(self, style='Root.TFrame')
        self.container.pack(fill="both", expand=True)

        self.login_frame = LoginFrame(self.container, self)
        self.register_frame = RegisterFrame(self.container, self)
        self.main_frame = MainFrame(self.container, self)

        self.show_login()

    def _setup_styles(self):
        """Configure modern styling"""
        style = ttk.Style(self)
        
        # Try to use a modern theme
        try:
            style.theme_use('clam')
        except:
            pass
        
        # Configure colors and styles
        style.configure('Root.TFrame', background=COLORS['bg'])
        style.configure('Main.TFrame', background=COLORS['bg'])
        style.configure('Card.TFrame', background=COLORS['white'], relief='flat')
        style.configure('Panel.TFrame', background=COLORS['white'], relief='flat')
        style.configure('Header.TFrame', background=COLORS['bg'])
        style.configure('Content.TFrame', background=COLORS['bg'])
        style.configure('Form.TFrame', background=COLORS['white'])
        style.configure('Action.TFrame', background=COLORS['white'])
        style.configure('Output.TFrame', background=COLORS['white'], relief='solid', borderwidth=1)
        
        # Labels
        style.configure('Title.TLabel', background=COLORS['white'], foreground=COLORS['primary'], 
                       font=("Segoe UI", 24, "bold"))
        style.configure('Subtitle.TLabel', background=COLORS['white'], foreground=COLORS['light_text'])
        style.configure('SectionTitle.TLabel', background=COLORS['white'], foreground=COLORS['primary'])
        style.configure('FieldLabel.TLabel', background=COLORS['white'], foreground=COLORS['text'])
        style.configure('Status.TLabel', background=COLORS['white'], foreground=COLORS['light_text'])
        
        # Buttons
        style.configure('Primary.TButton', 
                       background=COLORS['accent'], foreground='white', 
                       borderwidth=0, focuscolor='none', font=("Segoe UI", 10, "bold"))
        style.map('Primary.TButton',
                 background=[('active', '#4C6A8F'), ('pressed', '#3E5A7D')])
        
        style.configure('Secondary.TButton', 
                       background=COLORS['border'], foreground=COLORS['text'], 
                       borderwidth=0, focuscolor='none', font=("Segoe UI", 9))
        style.map('Secondary.TButton',
                 background=[('active', '#BDC4D1'), ('pressed', '#A8B1BE')])
        
        style.configure('Link.TButton', 
                       background=COLORS['white'], foreground=COLORS['accent'], 
                       borderwidth=0, focuscolor='none', font=("Segoe UI", 9))
        style.map('Link.TButton',
                 foreground=[('active', '#4C6A8F'), ('pressed', '#3E5A7D')])
        
        # Entries
        style.configure('Custom.TEntry', 
                       background=COLORS['white'], foreground=COLORS['text'],
                       borderwidth=1, relief='solid', insertcolor=COLORS['text'])
        style.map('Custom.TEntry',
                 bordercolor=[('focus', COLORS['accent'])])
        
        style.configure('Path.TEntry', 
                       background=COLORS['bg'], foreground=COLORS['light_text'],
                       borderwidth=1, relief='solid')
        
        # Combobox
        style.configure('Custom.TCombobox', 
                       background=COLORS['white'], foreground=COLORS['text'],
                       borderwidth=1, relief='solid')
        
        # Radiobuttons
        style.configure('Custom.TRadiobutton', 
                       background=COLORS['white'], foreground=COLORS['text'],
                       focuscolor='none')
        
        # LabelFrame
        style.configure('Section.TLabelframe', 
                       background=COLORS['white'], borderwidth=1, relief='solid')
        style.configure('Section.TLabelframe.Label', 
                       background=COLORS['white'], foreground=COLORS['primary'],
                       font=("Segoe UI", 11, "bold"))
        
        # Progress bar
        style.configure('Custom.Horizontal.TProgressbar',
                       background=COLORS['accent'], troughcolor=COLORS['border'],
                       borderwidth=0, lightcolor=COLORS['accent'], darkcolor=COLORS['accent'])

    # Navigation methods remain the same
    def _clear(self):
        for w in self.container.winfo_children():
            w.pack_forget()

    def show_login(self):
        self._clear()
        self.login_frame.pack(fill="both", expand=True)

    def show_register(self):
        self._clear()
        self.register_frame.pack(fill="both", expand=True)

    def show_main(self):
        self._clear()
        self.main_frame.pack(fill="both", expand=True)

    def logout(self):
        self.current_user = None
        self.show_login()

# -------------------- Main --------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()