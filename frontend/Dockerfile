# Build context: frontend/ (docker build -t voice-translator-frontend frontend)
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html vite.config.ts tsconfig*.json ./
COPY src ./src
COPY public ./public
# Always build the deployed image against the real API.
ENV VITE_TRANSLATION_MOCK=false
RUN npm run build

FROM nginx:stable-alpine AS serve
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
