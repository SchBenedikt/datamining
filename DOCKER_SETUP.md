# Docker Setup Guide

This guide explains how to run the unified crawler system using Docker and Docker Compose.

## Prerequisites

- Docker installed (version 20.10 or higher)
- Docker Compose installed (version 2.0 or higher)
- A configured `.env` file (see below)

## Configuration

### 1. Create `.env` File

Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=5432

# Email Notifications
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_email_password
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
ALERT_EMAIL=alert_recipient@example.com

# Discord Bot
DISCORD_TOKEN=your_discord_bot_token
CHANNEL_ID=your_discord_channel_id

# Google AI (for Streamlit dashboard)
GOOGLE_API_KEY=your_google_api_key
```

## Running with Docker Compose

### Start All Services

```bash
docker-compose up -d
```

This will start:
- Heise Archive Crawler
- Heise Live Crawler (every 5 minutes)
- Chip Archive Crawler
- Chip Live Crawler (every 10 minutes)
- Streamlit Dashboard (accessible at http://localhost:8501)
- Discord Bot

### View Logs

View logs for all services:
```bash
docker-compose logs -f
```

View logs for a specific service:
```bash
docker-compose logs -f heise-live-crawler
```

### Stop All Services

```bash
docker-compose down
```

### Restart a Specific Service

```bash
docker-compose restart heise-live-crawler
```

### Stop a Specific Service

```bash
docker-compose stop chip-archive-crawler
```

## Individual Service Management

### Start Only Specific Services

Start only the live crawlers and dashboard:
```bash
docker-compose up -d heise-live-crawler chip-live-crawler streamlit-dashboard
```

### Scale Services (if needed)

You cannot scale these services as they maintain state in the database.

## Accessing the Dashboard

Once the services are running:
- Streamlit Dashboard: http://localhost:8501

## Monitoring

### Check Service Status

```bash
docker-compose ps
```

### View Resource Usage

```bash
docker stats
```

## Troubleshooting

### Issue: Services fail to start
**Solution**: Check logs with `docker-compose logs [service-name]`

### Issue: Database connection errors
**Solution**: Verify `.env` file has correct database credentials and the database is accessible from Docker

### Issue: Dashboard not accessible
**Solution**: 
1. Check if port 8501 is already in use
2. Modify `docker-compose.yml` to use a different port: `"8502:8501"`

### Issue: Crawlers running but not collecting data
**Solution**: Check logs for HTTP errors or database connection issues

## Advanced Configuration

### Custom Port for Dashboard

Edit `docker-compose.yml`:
```yaml
streamlit-dashboard:
  ports:
    - "8080:8501"  # Change 8080 to your preferred port
```

### Add Additional Services

You can add more services to `docker-compose.yml`:
```yaml
  api-server:
    build: .
    command: python3 heise/api.py
    ports:
      - "6600:6600"
    env_file:
      - .env
    restart: unless-stopped
```

### Persistent Logs

Add volumes for log persistence:
```yaml
volumes:
  - ./logs:/app/logs
```

## Best Practices

1. **Use environment variables** for all sensitive data
2. **Monitor logs regularly** to catch issues early
3. **Backup your database** regularly
4. **Use Docker networks** to isolate services
5. **Set resource limits** if running on limited hardware

## Production Deployment

For production deployment:

1. Use a reverse proxy (nginx) for the Streamlit dashboard
2. Set up proper logging with log rotation
3. Use Docker secrets instead of environment variables
4. Implement health checks
5. Set up monitoring and alerting

Example health check in `docker-compose.yml`:
```yaml
heise-live-crawler:
  healthcheck:
    test: ["CMD", "python3", "-c", "import psycopg2; psycopg2.connect('${DB_CONNECTION_STRING}')"]
    interval: 30s
    timeout: 10s
    retries: 3
```

## Stopping and Cleaning Up

### Stop all services and remove containers
```bash
docker-compose down
```

### Stop and remove containers, networks, and volumes
```bash
docker-compose down -v
```

### Remove all images
```bash
docker-compose down --rmi all
```

## Support

For issues specific to Docker setup:
1. Check Docker logs: `docker-compose logs`
2. Verify network connectivity: `docker network inspect crawler-network`
3. Check container status: `docker-compose ps`
4. Review environment variables: `docker-compose config`
