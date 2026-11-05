# Pinned unprivileged Nginx image
FROM nginxinc/nginx-unprivileged:1.27-alpine

# Copy custom non-root nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy static chat UI
COPY index.html /usr/share/nginx/html/index.html

# Expose non-privileged port
EXPOSE 8080

USER 101

# Exec-form startup command
CMD ["nginx", "-g", "daemon off;"]
