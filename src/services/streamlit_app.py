import streamlit as st
import requests
import json
import time
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def get_full_image_url(relative_url):
    """将相对URL转换为完整URL"""
    if relative_url.startswith("http"):
        return relative_url
    if relative_url.startswith("/"):
        return f"{API_BASE_URL}{relative_url}"
    return f"{API_BASE_URL}/{relative_url}"

st.set_page_config(
    page_title="天文Agent测试助手",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 天文Agent测试助手")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_processed" not in st.session_state:
    st.session_state.image_processed = False

def parse_sse_stream(response):
    """解析SSE流"""
    buffer = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                try:
                    event = json.loads(data)
                    yield event
                except json.JSONDecodeError:
                    pass

def send_query(query, image_file=None, audio_file=None):
    """发送查询请求"""
    if audio_file is not None:
        files = {"audio": audio_file}
        data = {"query": query}
        response = requests.post(
            f"{API_BASE_URL}/query_with_audio",
            files=files,
            data=data,
            stream=True
        )
    elif image_file is not None:
        files = {"image": image_file}
        data = {"query": query}
        response = requests.post(
            f"{API_BASE_URL}/query_with_image",
            files=files,
            data=data,
            stream=True
        )
    else:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query},
            stream=True
        )
    
    if response.status_code == 200:
        return parse_sse_stream(response)
    else:
        st.error(f"请求失败: {response.text}")
        return None

with st.sidebar:
    st.header("⚙️ 控制面板")
    
    if st.button("🗑️ 清空记忆"):
        try:
            resp = requests.post(f"{API_BASE_URL}/clear_memory")
            if resp.status_code == 200:
                st.session_state.messages = []
                st.success("记忆已清空!")
            else:
                st.error("清空记忆失败")
        except Exception as e:
            st.error(f"连接失败: {e}")
    
    st.divider()
    
    st.subheader("💡 使用说明")
    st.markdown("""
    1. 在下方输入你的问题
    2. 可选择上传图片进行多模态问答
    3. 可选择录制语音或上传音频进行语音问答
    4. 点击发送按钮获取流式回答
    5. 回答会实时显示在界面上
    """)

tab1, tab2, tab3 = st.tabs(["💬 文本问答", "🖼️ 图片问答", "🎙️ 语音问答"])

with tab1:
    query_input = st.text_area(
        "输入你的天文问题:",
        height=100,
        placeholder="例如: 今晚北京能看到哪些行星？",
        key="query_text"
    )
    
    if st.button("🚀 发送", key="send_text"):
        if query_input.strip():
            st.session_state.messages.append({
                "role": "user",
                "content": query_input,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            
            with st.spinner("Agent正在思考中..."):
                try:
                    events = send_query(query_input)
                    if events:
                        assistant_message = {"role": "assistant", "content": "", "time": datetime.now().strftime("%H:%M:%S")}
                        thinking_placeholder = st.empty()
                        response_placeholder = st.empty()
                        thinking_content = ""
                        
                        for event in events:
                            event_type = event.get("type")
                            if event_type == "thinking":
                                thinking_content += event.get("content", "")
                                thinking_placeholder.markdown(f"🤔 *思考中:* {thinking_content}")
                            elif event_type == "text":
                                if thinking_content:
                                    thinking_placeholder.empty()
                                    thinking_content = ""
                                assistant_message["content"] += event.get("content", "")
                                response_placeholder.markdown(assistant_message["content"])
                            elif event_type == "image":
                                meta = event.get("meta", {})
                                image_url = get_full_image_url(event.get("url"))
                                if meta.get("source") == "user_upload":
                                    st.image(image_url, caption="您上传的图片")
                                else:
                                    st.image(image_url, caption="返回的图片")
                            
                        st.session_state.messages.append(assistant_message)
                except Exception as e:
                    st.error(f"错误: {e}")
        else:
            st.warning("请输入问题")

with tab2:
    query_with_image = st.text_area(
        "输入关于图片的问题:",
        height=100,
        placeholder="例如: 这张图片中的星座是什么？",
        key="query_image_text"
    )
    
    uploaded_file = st.file_uploader(
        "上传图片",
        type=['png', 'jpg', 'jpeg', 'gif', 'webp']
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="上传的图片", use_container_width=True)
    
    if st.button("🚀 发送", key="send_image"):
        if query_with_image.strip() and uploaded_file is not None:
            st.session_state.messages.append({
                "role": "user",
                "content": f"[图片] {query_with_image}",
                "time": datetime.now().strftime("%H:%M:%S")
            })
            
            with st.spinner("Agent正在分析图片..."):
                try:
                    events = send_query(query_with_image, uploaded_file)
                    if events:
                        assistant_message = {"role": "assistant", "content": "", "time": datetime.now().strftime("%H:%M:%S")}
                        thinking_placeholder = st.empty()
                        response_placeholder = st.empty()
                        thinking_content = ""
                        
                        for event in events:
                            event_type = event.get("type")
                            if event_type == "thinking":
                                thinking_content += event.get("content", "")
                                thinking_placeholder.markdown(f"🤔 *思考中:* {thinking_content}")
                            elif event_type == "text":
                                if thinking_content:
                                    thinking_placeholder.empty()
                                    thinking_content = ""
                                assistant_message["content"] += event.get("content", "")
                                response_placeholder.markdown(assistant_message["content"])
                            elif event_type == "image":
                                meta = event.get("meta", {})
                                image_url = get_full_image_url(event.get("url"))
                                if meta.get("source") == "user_upload":
                                    st.image(image_url, caption="您上传的图片")
                                else:
                                    st.image(image_url, caption="返回的图片")
                        
                        st.session_state.messages.append(assistant_message)
                except Exception as e:
                    st.error(f"错误: {e}")
        else:
            st.warning("请输入问题并上传图片")

with tab3:
    query_with_audio = st.text_area(
        "补充文本问题（可选）:",
        height=100,
        placeholder="例如: 请回答我语音中的天文问题",
        key="query_audio_text"
    )

    recorded_audio = st.audio_input("录制语音:", key="audio_recorder")

    uploaded_audio = st.file_uploader(
        "或上传音频文件",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        key="audio_file_upload"
    )

    # 优先使用录音，其次使用上传的音频文件
    audio_source = recorded_audio if recorded_audio is not None else uploaded_audio

    if audio_source is not None:
        st.audio(audio_source)

    if st.button("🚀 发送", key="send_audio"):
        if audio_source is not None:
            st.session_state.messages.append({
                "role": "user",
                "content": f"[语音] {query_with_audio}" if query_with_audio.strip() else "[语音输入]",
                "time": datetime.now().strftime("%H:%M:%S")
            })

            with st.spinner("Agent正在识别语音并思考中..."):
                try:
                    events = send_query(query_with_audio, audio_file=audio_source)
                    if events:
                        assistant_message = {"role": "assistant", "content": "", "time": datetime.now().strftime("%H:%M:%S")}
                        transcription_placeholder = st.empty()
                        thinking_placeholder = st.empty()
                        response_placeholder = st.empty()
                        thinking_content = ""

                        for event in events:
                            event_type = event.get("type")
                            if event_type == "transcription":
                                transcription_placeholder.info(f"🎙️ 语音识别结果: {event.get('text', '')}")
                            elif event_type == "thinking":
                                thinking_content += event.get("content", "")
                                thinking_placeholder.markdown(f"🤔 *思考中:* {thinking_content}")
                            elif event_type == "text":
                                if thinking_content:
                                    thinking_placeholder.empty()
                                    thinking_content = ""
                                assistant_message["content"] += event.get("content", "")
                                response_placeholder.markdown(assistant_message["content"])
                            elif event_type == "image":
                                meta = event.get("meta", {})
                                image_url = get_full_image_url(event.get("url"))
                                if meta.get("source") == "user_upload":
                                    st.image(image_url, caption="您上传的图片")
                                else:
                                    st.image(image_url, caption="返回的图片")

                        st.session_state.messages.append(assistant_message)
                except Exception as e:
                    st.error(f"错误: {e}")
        else:
            st.warning("请录制语音或上传音频文件")

st.divider()

st.subheader("📝 对话历史")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.caption(f"⏰ {msg['time']}")
