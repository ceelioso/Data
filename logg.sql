CREATE TABLESPACE PYTHONLOGIN_DATA
    DATAFILE 'pythonlogin_data.dbf'
    SIZE 50M
    AUTOEXTEND ON
    NEXT 10M
    MAXSIZE UNLIMITED;

-- Create user (if necessary)
CREATE USER PYTHONLOGIN IDENTIFIED BY password DEFAULT TABLESPACE PYTHONLOGIN_DATA;

-- Grant necessary privileges to the user
GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE TO PYTHONLOGIN;

-- Connect to the database schema
CONNECT pythonlogin/password;

-- Create table
CREATE TABLE ACCOUNTS (
    ID NUMBER(11) PRIMARY KEY,
    USERNAME VARCHAR2(50) NOT NULL,
    PASSWORD VARCHAR2(255) NOT NULL,
    EMAIL VARCHAR2(100) NOT NULL
);

-- Insert data into the table
INSERT INTO ACCOUNTS (
    ID,
    USERNAME,
    PASSWORD,
    EMAIL
) VALUES (
    1,
    'test',
    '0ef15de6149819f2d10fc25b8c994b574245f193',
    'test@test.com'
);