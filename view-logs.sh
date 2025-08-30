#!/bin/bash

# Script to view different types of logs

echo "=== Local LLM Interface Log Viewer ==="
echo ""

# Check if logs directory exists
if [ ! -d "./logs" ]; then
    echo "Creating logs directory..."
    mkdir -p logs
fi

case "${1:-help}" in
    "app"|"application")
        echo "📋 Viewing Application Logs (from file):"
        echo "=========================================="
        if [ -f "./logs/app.log" ]; then
            tail -f ./logs/app.log
        else
            echo "❌ Application log file not found. Container may not be running."
            echo "💡 Try: docker-compose up -d"
        fi
        ;;
    
    "docker"|"container")
        echo "🐳 Viewing Docker Container Logs:"
        echo "=================================="
        docker logs -f local-llm-api
        ;;
    
    "all"|"both")
        echo "📊 Viewing Both Application and Docker Logs:"
        echo "============================================="
        echo ""
        echo "🔹 Application logs (last 20 lines):"
        if [ -f "./logs/app.log" ]; then
            tail -20 ./logs/app.log
        else
            echo "❌ Application log file not found."
        fi
        echo ""
        echo "🔹 Docker logs (last 20 lines):"
        docker logs --tail 20 local-llm-api
        ;;
    
    "tail"|"follow")
        echo "👀 Following Application Logs in Real-time:"
        echo "==========================================="
        if [ -f "./logs/app.log" ]; then
            tail -f ./logs/app.log
        else
            echo "❌ Application log file not found."
            echo "💡 Starting container logs instead:"
            docker logs -f local-llm-api
        fi
        ;;
    
    "errors"|"error")
        echo "🚨 Filtering Error Messages:"
        echo "============================"
        if [ -f "./logs/app.log" ]; then
            grep -i "error\|warning\|exception\|failed" ./logs/app.log | tail -20
        else
            echo "❌ Application log file not found."
            echo "💡 Checking docker logs for errors:"
            docker logs local-llm-api 2>&1 | grep -i "error\|warning\|exception\|failed" | tail -20
        fi
        ;;
    
    "clear"|"clean")
        echo "🧹 Clearing Log Files:"
        echo "======================"
        if [ -f "./logs/app.log" ]; then
            > ./logs/app.log
            echo "✅ Application log cleared"
        fi
        echo "💡 To clear Docker logs, restart the container: docker-compose restart"
        ;;
    
    "help"|*)
        echo "📖 Usage: ./view-logs.sh [option]"
        echo ""
        echo "Options:"
        echo "  app, application  - View application logs from file"
        echo "  docker, container - View Docker container logs" 
        echo "  all, both        - View both application and Docker logs"
        echo "  tail, follow     - Follow application logs in real-time"
        echo "  errors, error    - Filter and show error messages"
        echo "  clear, clean     - Clear application log file"
        echo "  help             - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./view-logs.sh app       # View application logs"
        echo "  ./view-logs.sh follow    # Follow logs in real-time"
        echo "  ./view-logs.sh errors    # Show only errors"
        ;;
esac