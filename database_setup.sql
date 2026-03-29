-- ===============================
-- VoiceScribeAI Database Setup
-- ===============================

-- 1. Create Database
CREATE DATABASE IF NOT EXISTS whisper_app;

-- 2. Use Database
USE whisper_app;

-- ===============================
-- 3. Users Table
-- ===============================

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(128) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===============================
-- 4. Sample User (optional)
-- ===============================

-- Password is plain here (your app hashes it internally)
INSERT INTO users (username, password)
VALUES ('admin', 'admin123');

-- ===============================
-- END
-- ===============================
