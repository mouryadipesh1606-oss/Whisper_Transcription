import os
import whisper
import time
from datetime import datetime
import re
from collections import defaultdict

# For voice-based speaker detection (install required)
try:
    import librosa
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cosine
    VOICE_ANALYSIS_AVAILABLE = True
    print("🎵 Voice analysis libraries loaded successfully!")
except ImportError:
    VOICE_ANALYSIS_AVAILABLE = False
    print("⚠️ Voice analysis libraries not found. Install: pip install librosa scikit-learn")
    print("📝 Falling back to basic speaker detection")

def generate_output_filename(input_file, output_folder):
    """Generate filename same as input file but with .txt extension"""
    
    # Get original filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create filename with .txt extension
    filename = f"{base_name}.txt"
    
    # Full path
    full_path = os.path.join(output_folder, filename)
    
    return full_path

def format_time(seconds):
    """Convert seconds to readable time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

# Name detection function removed - using only Speaker 1, 2, 3, etc.

def extract_voice_features(audio_file, segments):
    """Extract voice features for each segment to identify different speakers"""
    if not VOICE_ANALYSIS_AVAILABLE:
        return None
    
    try:
        print("🎤 Analyzing voice patterns...")
        # Load audio file
        y, sr = librosa.load(audio_file, sr=16000)
        
        voice_features = []
        
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            # Extract audio segment
            segment_audio = y[start_sample:end_sample]
            
            if len(segment_audio) < sr * 0.5:  # Skip very short segments
                voice_features.append(None)
                continue
            
            # Extract MFCC features (voice characteristics)
            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            
            # Calculate mean and std of MFCCs
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Extract pitch/fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Combine features
            features = np.concatenate([mfcc_mean, mfcc_std, [pitch_mean]])
            voice_features.append(features)
        
        return voice_features
    
    except Exception as e:
        print(f"❌ Voice analysis error: {e}")
        return None

def cluster_speakers_by_voice(voice_features, max_speakers=4):
    """Cluster segments by voice characteristics"""
    if voice_features is None or not VOICE_ANALYSIS_AVAILABLE:
        return None
    
    # Filter out None features
    valid_features = []
    valid_indices = []
    
    for i, features in enumerate(voice_features):
        if features is not None:
            valid_features.append(features)
            valid_indices.append(i)
    
    if len(valid_features) < 2:
        return None
    
    try:
        # Normalize features
        features_array = np.array(valid_features)
        
        # Determine optimal number of speakers (2-4)
        n_speakers = min(max_speakers, max(2, len(valid_features) // 10))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_array)
        
        # Map back to all segments
        speaker_assignments = [0] * len(voice_features)  # Default to speaker 0
        
        for i, valid_idx in enumerate(valid_indices):
            speaker_assignments[valid_idx] = cluster_labels[i] + 1  # 1-based numbering
        
        print(f"🎯 Detected {n_speakers} different speakers by voice analysis")
        return speaker_assignments
    
    except Exception as e:
        print(f"❌ Clustering error: {e}")
        return None

def detect_speaker_changes(segments, audio_file=None, use_voice_analysis=True):
    """Advanced speaker detection using voice analysis or fallback to basic detection"""
    if not segments:
        return segments
    
    speakers_text = []
    
    # Try voice-based detection first
    if use_voice_analysis and VOICE_ANALYSIS_AVAILABLE and audio_file:
        print("🎵 Using voice-based speaker detection...")
        
        voice_features = extract_voice_features(audio_file, segments)
        speaker_assignments = cluster_speakers_by_voice(voice_features)
        
        if speaker_assignments:
            # Use voice-based assignments
            for i, segment in enumerate(segments):
                speaker_num = speaker_assignments[i] if i < len(speaker_assignments) else 1
                speaker_label = f"Speaker {speaker_num}"
                
                speakers_text.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'speaker': speaker_label
                })
            
            return speakers_text
    
    # Fallback to basic pause-based detection
    print("🎤 Using basic pause-based speaker detection...")
    
    current_speaker = 1
    max_speakers = 4  # Limit to 4 speakers for basic detection
    
    for i, segment in enumerate(segments):
        text = segment['text'].strip()
        
        # Simple heuristics for speaker change (conservative approach)
        speaker_change = False
        
        if i > 0:
            prev_end = segments[i-1]['end']
            current_start = segment['start']
            pause_duration = current_start - prev_end
            
            # Only change speaker on very long pauses (4+ seconds)
            if pause_duration > 4.0:
                speaker_change = True
            
            # Or on very clear conversation markers with medium pause
            clear_markers = [
                'hello', 'hi there', 'yes sir', 'no sir', 'thank you very much',
                'excuse me', 'i think', 'well then', 'okay so',
                'हैलो', 'नमस्ते', 'हां जी हां', 'नहीं जी', 'धन्यवाद', 'माफ करिए'
            ]
            
            text_lower = text.lower()
            if any(text_lower.startswith(marker) for marker in clear_markers):
                if pause_duration > 2.0:
                    speaker_change = True
        
        if speaker_change:
            # Cycle through speakers
            current_speaker = (current_speaker % max_speakers) + 1
        
        speaker_label = f"Speaker {current_speaker}"
        
        speakers_text.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': text,
            'speaker': speaker_label
        })
    
    return speakers_text

def format_transcript_with_speakers(segments_with_speakers):
    """Format transcript with speaker labels and timestamps"""
    formatted_text = ""
    
    for segment in segments_with_speakers:
        timestamp = format_time(segment['start'])
        speaker = segment['speaker']
        text = segment['text']
        
        formatted_text += f"[{timestamp}] {speaker}: {text}\n\n"
    
    return formatted_text

def main():
    print("=== Enhanced Audio Transcription with Speaker Detection ===")
    print("🎤 Detects speakers and identifies names automatically!")
    print("👨‍💻 Developed by Aditi Pandit")
    print("📁 Auto-generates filename based on audio file!")
    print()
    
    # Get input audio file
    input_path = input("🎵 Enter audio file path: ").strip().strip('"')
    
    if not os.path.isfile(input_path):
        print('❌ Error: Audio file not found.')
        return
    
    # Get output folder (not file!)
    output_folder = input("📂 Enter output folder path (where to save): ").strip().strip('"')
    
    # Create folder if it doesn't exist
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"✅ Created folder: {output_folder}")
        except:
            print("❌ Could not create folder. Using current directory.")
            output_folder = "."
    
    # Generate automatic filename
    output_file = generate_output_filename(input_path, output_folder)
    
    print(f"📝 Output filename: {os.path.basename(output_file)}")
    
    # Model selection
    print("\n🤖 Model options:")
    print("1. tiny - Fastest (recommended for long files)")
    print("2. base - Balanced") 
    print("3. small - Better accuracy")
    print("4. medium - High accuracy (slower)")
    print("5. large - Best accuracy (very slow)")
    
    model_choice = input("Choose model (1/2/3/4/5) [default: 1]: ").strip()
    model_names = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    model_name = model_names.get(model_choice, "tiny")
    
    # Language selection
    print("\n🌐 Language options:")
    print("Common codes: en (English), hi (Hindi), fr (French), es (Spanish),")
    print("             de (German), ja (Japanese), zh (Chinese), ar (Arabic),")
    print("             ru (Russian), pt (Portuguese), it (Italian), ko (Korean)")
    language = input("Enter language code (or 'auto' for detection) [default: auto]: ").strip()
    if language.lower() in ['auto', '']:
        language = None
    
    # Speaker detection option
    print("\n🎤 Speaker Detection Options:")
    print("1. Voice-based (Advanced - analyzes actual voice patterns)")
    print("2. Basic (Simple pause-based detection)")
    print("3. Disabled")
    
    speaker_choice = input("Choose option (1/2/3) [default: 1]: ").strip()
    
    if speaker_choice == '3':
        enable_speakers = False
        use_voice_analysis = False
    elif speaker_choice == '2':
        enable_speakers = True
        use_voice_analysis = False
    else:  # Default to voice-based
        enable_speakers = True
        use_voice_analysis = True
    
    try:
        print(f"\n🔄 Loading {model_name} model...")
        model = whisper.load_model(model_name)
        
        print("⏳ Processing audio... (this may take a while)")
        print("💡 Progress will be shown below:")
        print("-" * 60)
        
        start_time = time.time()
        
        # Transcribe with word timestamps for better speaker detection
        if language:
            result = model.transcribe(
                input_path, 
                language=language,
                fp16=False,
                verbose=True,
                word_timestamps=True if enable_speakers else False,
                temperature=0.0,
            )
        else:
            result = model.transcribe(
                input_path,
                fp16=False,
                verbose=True,
                word_timestamps=True if enable_speakers else False,
                temperature=0.0,
            )
        
        processing_time = time.time() - start_time
        text = result["text"].strip()
        
        # Process speakers if enabled
        speaker_transcript = ""
        
        if enable_speakers and 'segments' in result:
            print("\n🔍 Analyzing speaker changes...")
            
            # Speaker detection with voice analysis option
            segments_with_speakers = detect_speaker_changes(
                result['segments'], 
                audio_file=input_path if use_voice_analysis else None,
                use_voice_analysis=use_voice_analysis
            )
            
            # Format transcript with speakers
            speaker_transcript = format_transcript_with_speakers(segments_with_speakers)
        
        # Create detailed output content
        output_content = f"""ENHANCED AUDIO TRANSCRIPTION REPORT
{"="*60}
👨‍💻 Developed by: Aditi Pandit
{"="*60}

📁 Original File: {os.path.basename(input_path)}
📅 Transcription Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏱️  Processing Time: {format_time(processing_time)}
🌐 Detected Language: {result.get('language', 'Unknown').upper()}
🤖 Model Used: {model_name.upper()}
📊 Text Length: {len(text)} characters
📏 Estimated Words: {len(text.split())} words
🎤 Speaker Detection: {'Enabled' if enable_speakers else 'Disabled'}

{"="*60}
FULL TRANSCRIPT (Plain Text):
{"="*60}

{text}

"""

        if enable_speakers and speaker_transcript:
            output_content += f"""
{"="*60}
SPEAKER-SEPARATED TRANSCRIPT:
{"="*60}

{speaker_transcript}
"""

        output_content += f"""
{"="*60}
END OF TRANSCRIPT
{"="*60}

🎵 Generated by Enhanced Whisper Speech-to-Text
👨‍💻 Developed by Aditi Pandit
📁 Filename: {os.path.basename(output_file)}
⚡ Features: Auto-transcription, Speaker detection, Name recognition
"""
        
        # Save to auto-generated file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print("\n" + "🎉" * 25)
        print("✅ ENHANCED TRANSCRIPTION COMPLETED SUCCESSFULLY!")
        print("🎉" * 25)
        print(f"📁 Saved to: {output_file}")
        print(f"⏱️  Processing time: {format_time(processing_time)}")
        print(f"🌐 Language: {result.get('language', 'Unknown').upper()}")
        print(f"📊 Characters: {len(text)}")
        print(f"📝 Words: {len(text.split())}")
        print(f"🎤 Speaker Detection: {'✅ Enabled' if enable_speakers else '❌ Disabled'}")
        
        print(f"\n📖 Preview (first 200 characters):")
        print("-" * 50)
        print(text[:200] + "..." if len(text) > 200 else text)
        print("-" * 50)
        print("\n👨‍💻 Developed by Aditi Pandit")
        
    except KeyboardInterrupt:
        print("\n⏹️  [Transcription stopped by user]")
    except Exception as e:
        print(f'\n❌ Error: {e}')
        print("\n💡 Tips:")
        print("• Use 'tiny' model for large files")
        print("• Enable speaker detection for conversations")
        print("• Ensure folder path exists and has write permissions")
        print("• Check available disk space")

if __name__ == '__main__':
    main()