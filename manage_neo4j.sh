#!/bin/bash

# Create necessary directories
mkdir -p neo4j/data neo4j/logs neo4j/conf

case "$1" in
  start)
    echo "Starting Neo4j container..."
    docker-compose up -d
    echo "Neo4j is running!"
    echo "- Browser interface: http://localhost:7474"
    echo "- Login credentials: neo4j/saga_password"
    ;;
    
  stop)
    echo "Stopping Neo4j container..."
    docker-compose down
    ;;
    
  status)
    if docker-compose ps | grep -q "neo4j.*Up"; then
      echo "Neo4j is running"
      echo "- Browser interface: http://localhost:7474"
    else
      echo "Neo4j is not running"
    fi
    ;;
    
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
