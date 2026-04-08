#!/usr/bin/env bash
# AstroAgent 一键启动脚本
# 启动顺序: MCP Server → API Backend → Streamlit Frontend
#./start.sh          # 启动所有服务（默认）
# ./start.sh stop     # 停止所有服务
# ./start.sh restart  # 重启所有服务
# ./start.sh status   # 查看运行状态

 set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/.pids"
SSL_DIR="$PROJECT_DIR/ssl"

mkdir -p "$LOG_DIR" "$PID_DIR"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

# 检查端口是否被占用
check_port() {
    if ss -tlnp 2>/dev/null | grep -q ":$1 "; then
        return 0  # 端口被占用
    fi
    return 1
}

# 停止已有服务
stop_services() {
    echo "========================================="
    echo "  停止 AstroAgent 所有服务"
    echo "========================================="
    for name in mcp_server api streamlit; do
        pid_file="$PID_DIR/${name}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
                wait "$pid" 2>/dev/null || true
                info "$name (PID $pid) 已停止"
            fi
            rm -f "$pid_file"
        fi
    done
    info "所有服务已停止"
}

# 启动所有服务
start_services() {
    echo ""
    echo "========================================="
    echo "  🌌 AstroAgent 一键启动"
    echo "========================================="
    echo ""

    cd "$PROJECT_DIR"

    # ---------- 1. MCP Server (port 8001) ----------
    if check_port 8001; then
        warn "端口 8001 已被占用，跳过 MCP Server 启动"
    else
        info "启动 MCP Server (port 8001) ..."
        nohup python3 -m src.services.mcp_server \
            > "$LOG_DIR/mcp_server.log" 2>&1 &
        echo $! > "$PID_DIR/mcp_server.pid"
        sleep 2
        if kill -0 "$(cat "$PID_DIR/mcp_server.pid")" 2>/dev/null; then
            info "MCP Server 已启动 (PID $(cat "$PID_DIR/mcp_server.pid"))"
        else
            error "MCP Server 启动失败，请检查 $LOG_DIR/mcp_server.log"
            exit 1
        fi
    fi

    # ---------- 2. API Backend (port 8000) ----------
    if check_port 8000; then
        warn "端口 8000 已被占用，跳过 API Backend 启动"
    else
        info "启动 API Backend (port 8000) ..."
        nohup python3 -m src.api.main \
            > "$LOG_DIR/api.log" 2>&1 &
        echo $! > "$PID_DIR/api.pid"
        sleep 3
        if kill -0 "$(cat "$PID_DIR/api.pid")" 2>/dev/null; then
            info "API Backend 已启动 (PID $(cat "$PID_DIR/api.pid"))"
        else
            error "API Backend 启动失败，请检查 $LOG_DIR/api.log"
            exit 1
        fi
    fi

    # ---------- 3. Streamlit Frontend (port 8501, HTTPS) ----------
    if check_port 8501; then
        warn "端口 8501 已被占用，跳过 Streamlit 启动"
    else
        # SSL 参数：有证书则用 HTTPS，否则 HTTP
        SSL_ARGS=""
        if [ -f "$SSL_DIR/cert.pem" ] && [ -f "$SSL_DIR/key.pem" ]; then
            SSL_ARGS="--server.sslCertFile=$SSL_DIR/cert.pem --server.sslKeyFile=$SSL_DIR/key.pem"
            info "启动 Streamlit Frontend (port 8501, HTTPS) ..."
        else
            warn "未找到 SSL 证书，以 HTTP 模式启动 (仅 localhost 可用录音)"
            info "启动 Streamlit Frontend (port 8501, HTTP) ..."
        fi

        nohup streamlit run src/services/streamlit_app.py \
            --server.address=0.0.0.0 \
            --server.port=8501 \
            --server.headless=true \
            $SSL_ARGS \
            > "$LOG_DIR/streamlit.log" 2>&1 &
        echo $! > "$PID_DIR/streamlit.pid"
        sleep 2
        if kill -0 "$(cat "$PID_DIR/streamlit.pid")" 2>/dev/null; then
            info "Streamlit Frontend 已启动 (PID $(cat "$PID_DIR/streamlit.pid"))"
        else
            error "Streamlit 启动失败，请检查 $LOG_DIR/streamlit.log"
            exit 1
        fi
    fi

    # ---------- 汇总 ----------
    echo ""
    echo "========================================="
    echo "  🚀 所有服务已启动"
    echo "========================================="
    echo ""
    echo "  MCP Server   : http://localhost:8001/mcp/"
    if [ -n "$SSL_ARGS" ]; then
        echo "  Streamlit     : https://localhost:8501"
        echo "  外部访问      : https://<公网IP>:8501"
    else
        echo "  Streamlit     : http://localhost:8501"
    fi
    echo "  API Backend   : http://localhost:8000"
    echo ""
    echo "  日志目录      : $LOG_DIR/"
    echo "  停止服务      : $0 stop"
    echo "  查看状态      : $0 status"
    echo "========================================="
}

# 查看服务状态
show_status() {
    echo "========================================="
    echo "  AstroAgent 服务状态"
    echo "========================================="
    for name in mcp_server api streamlit; do
        pid_file="$PID_DIR/${name}.pid"
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            info "$name 运行中 (PID $(cat "$pid_file"))"
        else
            error "$name 未运行"
        fi
    done
}

# 主入口
case "${1:-start}" in
    start)   stop_services 2>/dev/null; start_services ;;
    stop)    stop_services ;;
    restart) stop_services; sleep 1; start_services ;;
    status)  show_status ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
