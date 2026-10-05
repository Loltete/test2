#!/usr/bin/env bash
# รันในโฟลเดอร์โปรเจกต์ (หลังคัดลอกไฟล์ทั้งหมดมาแล้ว)
git init
git branch -M main
git add .
git status   # ตรวจว่าไม่มี .streamlit/secrets.toml อยู่ในรายการ
git commit -m "RAG chatbot: Prachinburi travel guide"
git remote add origin https://github.com/Loltete/test2.git
git push -u origin main
