# 화면 배포 이미지. 빌드 위치는 frontend/ 폴더다(docker build -t voice-translator-frontend frontend, 또는
# docker-compose.yml의 web). 첫 단계에서 Node로 정적 파일을 빌드하고, 둘째 단계의 nginx가 그 파일을 내주면서
# /api를 API 컨테이너로 넘긴다(nginx.conf).
# 두 기본 이미지는 태그와 함께 digest(sha256)로 고정한다. 태그만 쓰면 같은 Dockerfile이 빌드할 때마다 다른
# Node·nginx를 받을 수 있다. 2026-10-15의 node:24-alpine·nginx:stable-alpine(nginx 1.30)이다. 보안 수정을
# 받으려면 태그를 다시 받아 digest를 새로 적는다(backend/Dockerfile 머리 주석과 같은 명령).
FROM node:24-alpine@sha256:44fce696f38ba8c987c4db89954051000bbd05ffd8d8659ba15ed4a8602e937f AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html vite.config.ts tsconfig*.json ./
COPY src ./src
COPY public ./public
# 배포 이미지는 늘 실제 API를 쓰도록 빌드한다(예시 모드 끔).
ENV VITE_TRANSLATION_MOCK=false
RUN npm run build

FROM nginx:stable-alpine@sha256:ed84151ed2d47ef93c0939eb9b9754c6990db6092cd2081e11988a47a83d47ea AS serve
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
