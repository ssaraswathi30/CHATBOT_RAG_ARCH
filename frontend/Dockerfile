FROM nginx:alpine

# Copy custom nginx config from root
COPY nginx.conf /etc/nginx/nginx.conf

# Copy frontend files
COPY frontend/index.html /usr/share/nginx/html/

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
