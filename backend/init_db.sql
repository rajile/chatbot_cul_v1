-- Script de inicialización para TiDB Cloud
CREATE DATABASE IF NOT EXISTS cul_chatbot;
USE cul_chatbot;

CREATE TABLE IF NOT EXISTS chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_query TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
