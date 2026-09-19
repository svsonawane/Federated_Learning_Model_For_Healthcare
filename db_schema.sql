CREATE DATABASE IF NOT EXISTS federated_health;
USE federated_health;

CREATE TABLE IF NOT EXISTS patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120),
    age INT,
    gender VARCHAR(20),
    birthdate VARCHAR(20),
    weight_kg DECIMAL(5,2),
    height_cm DECIMAL(5,2),
    phone_number VARCHAR(20),
    address VARCHAR(255),
    emergency_contact_name VARCHAR(120),
    emergency_contact_relation VARCHAR(50),
    emergency_contact_phone VARCHAR(20),
    symptoms TEXT,
    past_conditions TEXT,
    family_history TEXT,
    diagnosis VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_weights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    round_number INT,
    weights_json LONGTEXT,
    meta_json LONGTEXT,
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
