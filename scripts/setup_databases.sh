#!/bin/bash

# Setup Databases Script
# This script initializes and configures databases for the AI/ML pipeline

set -e

echo "🔧 Setting up databases for AI/ML Pipeline..."

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed. Please install PostgreSQL first."
    exit 1
fi

# Check if Redis is installed
if ! command -v redis-cli &> /dev/null; then
    echo "❌ Redis is not installed. Please install Redis first."
    exit 1
fi

# Database configuration
DB_NAME="ai_ml_pipeline"
DB_USER="ai_ml_user"
DB_PASSWORD="ai_ml_password"
REDIS_PORT=6379

echo "📊 Setting up PostgreSQL database..."

# Create database and user
sudo -u postgres psql -c "DROP DATABASE IF EXISTS $DB_NAME;" || true
sudo -u postgres psql -c "DROP USER IF EXISTS $DB_USER;" || true
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

echo "✅ PostgreSQL database setup complete"

echo "🔴 Setting up Redis..."

# Start Redis service
if command -v systemctl &> /dev/null; then
    sudo systemctl start redis || sudo systemctl start redis-server
    sudo systemctl enable redis || sudo systemctl enable redis-server
else
    # For systems without systemd
    redis-server --daemonize yes --port $REDIS_PORT
fi

# Test Redis connection
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Redis setup complete"
else
    echo "❌ Redis connection failed"
    exit 1
fi

echo "🗄️ Running database migrations..."

# Run migrations (assuming you have migration scripts)
cd "$(dirname "$0")/.."

# Check if we have pixi available
if command -v pixi &> /dev/null; then
    pixi run python -c "
import sys
sys.path.append('projects/shared_utils/src')
from shared_utils.database import setup_database
setup_database()
print('Database migrations complete')
"
else
    echo "⚠️  Pixi not found, skipping migrations"
fi

echo "🎉 Database setup complete!"
echo ""
echo "Database connection details:"
echo "  PostgreSQL: localhost:5432/$DB_NAME"
echo "  Redis: localhost:$REDIS_PORT"
echo "  User: $DB_USER"
echo ""
echo "You can now start using the AI/ML Pipeline!"
