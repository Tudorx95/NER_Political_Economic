FROM node:20-alpine AS builder
WORKDIR /app

# Copiem doar package.json (deoarece yarn.lock lipsește)
COPY package.json ./

# Instalăm dependențele
RUN yarn install || npm install

# Copiem tot codul sursă
COPY . .

# Build aplicația React
RUN yarn build || npm run build

# === Production stage ===
FROM nginx:alpine

# 1. Stergem pagina implicita "Welcome to Nginx"
RUN rm -rf /usr/share/nginx/html/*

# 2. Copiem build-ul generat de React din etapa builder
COPY --from=builder /app/build /usr/share/nginx/html

# 3. Copiem configuratia ta care stie sa faca proxy spre backend
COPY ./nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]