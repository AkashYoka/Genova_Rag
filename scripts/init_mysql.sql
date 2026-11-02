-- Initialize MySQL 8.4 LTS database
-- This script runs once when the mysql container is first created.
-- The database itself is created from MYSQL_DATABASE (utf8mb4, see the
-- server flags in docker-compose.yml); this file is the place for any
-- additional setup the deployment needs.

SET NAMES utf8mb4;
