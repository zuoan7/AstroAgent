#!/usr/bin/env bash
# AstroAgent 一键启动脚本
# 启动顺序: MCP Server → API Backend → Streamlit Frontend
# ./start.sh          # 启动所有服务（默认）
# ./start.sh stop     # 停止所有服务
# ./start.sh restart  # 重启所有服务
# ./start.sh status   # 查看运行状态
# ./start.sh tunnel   # 创建 SSH 隧道（本地访问远程服务）

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -t -i :"$port" 2>/dev/null || true)

    if [ -n "$pids" ]; then
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
            fi
        done

        sleep 1
        pids=$(lsof -t -i :"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                kill -9 "$pid" 2>/dev/null || true
            done
            sleep 1
        fi
        return 0
    fi
    return 1
}

check_port() {
    local port=$1
    if ss -tlnp 2>/dev/null | grep -q ":$port "; then
        return 0
    fi
    return 1
}

stop_services() {
    echo "========================================="
    echo "  停止 AstroAgent 所有服务"
    echo "========================================="

    for name in mcp_server api streamlit; do
        local pid_file="$PID_DIR/${name}.pid"
        if [ -f "$pid_file" ]; then
            local pid
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                local count=0
                while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                    sleep 0.5
                    count=$((count + 1))
                done
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
                info "$name (PID $pid) 已停止"
            fi
            rm -f "$pid_file"
        fi
    done

    echo ""
    info "检查并释放残留端口..."
    for port in 8001 8000 8501; do
        if check_port "$port"; then
            warn "端口 $port 仍被占用，强制释放..."
            kill_port "$port"
            if check_port "$port"; then
                error "无法释放端口 $port"
            else
                info "端口 $port 已释放"
            fi
        fi
    done

    info "所有服务已停止"
}

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
        nohup env PYTHONPATH="$PROJECT_DIR" python3 "$PROJECT_DIR/src/services/mcp_server.py" \
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
        nohup env PYTHONPATH="$PROJECT_DIR" python3 "$PROJECT_DIR/src/api/main.py" \
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

    # ---------- 3. Streamlit Frontend (port 8501, HTTP only) ----------
    if check_port 8501; then
        warn "端口 8501 已被占用，尝试释放..."
        kill -9 $(lsof -t -i :8501) 2>/dev/null || true
        sleep 2
        if check_port 8501; then
            error "无法释放端口 8501，请手动处理"
            exit 1
        fi
    fi

    warn "已禁用 SSL，Streamlit 将以 HTTP 模式启动（适合 SSH 隧道访问）"
    info "启动 Streamlit Frontend (port 8501, HTTP) ..."

    nohup streamlit run src/services/streamlit_app.py \
        --server.address=0.0.0.0 \
        --server.port=8501 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --browser.serverAddress=localhost \
        --browser.gatherUsageStats=false \
        > "$LOG_DIR/streamlit.log" 2>&1 &
    echo $! > "$PID_DIR/streamlit.pid"
    sleep 3
    if kill -0 "$(cat "$PID_DIR/streamlit.pid")" 2>/dev/null; then
        info "Streamlit Frontend 已启动 (PID $(cat "$PID_DIR/streamlit.pid"))"
    else
        error "Streamlit 启动失败，请检查 $LOG_DIR/streamlit.log"
        exit 1
    fi

    # ---------- 汇总 ----------
    echo ""
    echo "========================================="
    echo "  🚀 所有服务已启动"
    echo "========================================="
    echo ""
    echo "  MCP Server   : http://localhost:8001/mcp/"
    echo "  Streamlit    : http://localhost:8501"
    echo "  API Backend  : http://localhost:8000"
    echo ""
    echo "  日志目录      : $LOG_DIR/"
    echo "  停止服务      : $0 stop"
    echo "  查看状态      : $0 status"
    echo "  创建隧道      : $0 tunnel <服务器地址> [本地端口]"
    echo "========================================="
}

show_status() {
    echo "========================================="
    echo "  AstroAgent 服务状态"
    echo "========================================="
    for name in mcp_server api streamlit; do
        local pid_file="$PID_DIR/${name}.pid"
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            info "$name 运行中 (PID $(cat "$pid_file"))"
        else
            error "$name 未运行"
        fi
    done
}

create_tunnel() {
    local server_host=$1
    local local_base_port=${2:-9000}

    if [ -z "$server_host" ]; then
        error "请指定服务器地址"
        echo "用法: $0 tunnel <服务器地址> [本地起始端口]"
        echo "示例: $0 tunnel user@example.com 9000"
        echo ""
        echo "这将创建以下隧道映射："
        echo "  本地 9000 -> 远程 8000 (API Backend)"
        echo "  本地 9001 -> 远程 8001 (MCP Server)"
        echo "  本地 9002 -> 远程 8501 (Streamlit)"
        exit 1
    fi

    local api_port=$((local_base_port))
    local mcp_port=$((local_base_port + 1))
    local streamlit_port=$((local_base_port + 2))

    echo "========================================="
    echo "  创建 SSH 隧道连接"
    echo "========================================="
    echo ""
    info "服务器: $server_host"
    echo ""
    echo "端口映射："
    echo "  本地 $api_port       -> 远程 8000 (API Backend)"
    echo "  本地 $mcp_port       -> 远程 8001 (MCP Server)"
    echo "  本地 $streamlit_port -> 远程 8501 (Streamlit)"
    echo ""
    warn "Streamlit 隧道注意事项："
    echo "  1. 请使用 HTTP 访问，不要使用 HTTPS"
    echo "  2. 浏览器访问: http://localhost:$streamlit_port"
    echo "  3. 如遇页面异常，先检查 logs/streamlit.log"
    echo ""
    info "正在建立隧道连接 (Ctrl+C 断开)..."
    echo ""

    ssh -N -L ${api_port}:localhost:8000 \
           -L ${mcp_port}:localhost:8001 \
           -L ${streamlit_port}:localhost:8501 \
           -o ServerAliveInterval=60 \
           -o ServerAliveCountMax=3 \
           -o ExitOnForwardFailure=yes \
           "$server_host"
}

case "${1:-start}" in
    start)   stop_services 2>/dev/null; start_services ;;
    stop)    stop_services ;;
    restart) stop_services; sleep 3; start_services ;;
    status)  show_status ;;
    tunnel)  create_tunnel "$2" "$3" ;;
    *)
        echo "用法: $0 {start|stop|restart|status|tunnel}"
        echo ""
        echo "命令说明："
        echo "  start   - 启动所有服务"
        echo "  stop    - 停止所有服务"
        echo "  restart - 重启所有服务"
        echo "  status  - 查看服务状态"
        echo "  tunnel  - 创建 SSH 隧道 (需指定服务器地址)"
        exit 1
        ;;
esac