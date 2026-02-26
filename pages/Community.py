import streamlit as st
import sqlite3
import uuid
import datetime
from streamlit_autorefresh import st_autorefresh

# ==============================
# 🌾 Page Config
# ==============================
st.set_page_config(
    page_title="Community Chat",
    page_icon="💬",
    layout="wide"
)

# ==============================
# 🔄 Auto Refresh Every 3 Seconds
# ==============================
st_autorefresh(interval=3000, key="chat_refresh")

# ==============================
# 🗄️ Database Setup
# ==============================
conn = sqlite3.connect("community.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    message TEXT,
    timestamp TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    user TEXT PRIMARY KEY,
    last_seen TEXT,
    typing INTEGER DEFAULT 0
)
""")

conn.commit()

# ==============================
# 👤 Anonymous Login
# ==============================
if "user_id" not in st.session_state:
    st.session_state.user_id = "Farmer-" + str(uuid.uuid4())[:6]

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

c.execute("""
INSERT INTO users (user, last_seen)
VALUES (?, ?)
ON CONFLICT(user) DO UPDATE SET last_seen=excluded.last_seen
""", (st.session_state.user_id, current_time))

conn.commit()

# ==============================
# 🧹 Remove Inactive Users (5 min timeout)
# ==============================
c.execute("""
DELETE FROM users
WHERE last_seen < datetime('now', '-5 minutes')
""")
conn.commit()

# ==============================
# 🎨 WhatsApp Style CSS
# ==============================
st.markdown("""
<style>
.chat-container {
    height: 4px;
    overflow-y: auto;
    padding: 15px;
    background-color: #00000;
    border-radius: 10px;
}

.message {
    padding: 10px 15px;
    border-radius: 20px;
    margin-bottom: 10px;
    max-width: 75%;
    word-wrap: break-word;
}

.me {
    background-color: #000000;
    margin-left: auto;
    text-align: right;
}

.other {
    background-color: #000000;
    margin-right: auto;
}

.user-list {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🌿 Layout
# ==============================
col1, col2 = st.columns([3, 1])

with col1:
    st.title("💬 Farmer Community")

with col2:
    st.subheader("🟢 Online")
    c.execute("SELECT user FROM users")
    online_users = c.fetchall()

    st.markdown("<div class='user-list'>", unsafe_allow_html=True)
    for user in online_users:
        st.write("🟢", user[0])
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ==============================
# 💬 Display Messages
# ==============================
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

c.execute("""
SELECT user, message FROM messages
ORDER BY id ASC
LIMIT 100
""")

rows = c.fetchall()

for user, message in rows:
    if user == st.session_state.user_id:
        st.markdown(
            f"<div class='message me'>{message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='message other'><b>{user}</b><br>{message}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# 💬 Typing Indicator
# ==============================
c.execute("""
SELECT user FROM users
WHERE typing=1 AND user != ?
""", (st.session_state.user_id,))
typing_users = c.fetchall()

if typing_users:
    for user in typing_users:
        st.caption(f"✍️ {user[0]} is typing...")

# ==============================
# 💬 Message Input
# ==============================
with st.form("chat_form", clear_on_submit=True):
    message = st.text_input("Type your message")

    # Update typing status
    if message:
        c.execute("""
        UPDATE users SET typing=1 WHERE user=?
        """, (st.session_state.user_id,))
    else:
        c.execute("""
        UPDATE users SET typing=0 WHERE user=?
        """, (st.session_state.user_id,))
    conn.commit()

    send = st.form_submit_button("Send")

    if send and message.strip():
        c.execute("""
        INSERT INTO messages (user, message, timestamp)
        VALUES (?, ?, ?)
        """, (st.session_state.user_id, message, current_time))

        c.execute("""
        UPDATE users SET typing=0 WHERE user=?
        """, (st.session_state.user_id,))

        conn.commit()
        st.rerun()
