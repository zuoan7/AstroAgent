#!/usr/bin/env bash
# AstroAgent 一键启动脚本
# 启动顺序: MCP Server → API Backend → Vue3 Frontend
# ./start.sh          # 启动所有服务（默认）
# ./start.sh stop     # 停止所有服务
# ./start.sh restart  # 重启所有服务
# ./start.sh status   # 查看运行状态
# ./start.sh tunnel   # 创建 SSH 隧道（本地访问远程服务）

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"
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

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    local cmd=$1
    local message=$2
    if ! command_exists "$cmd"; then
        error "$message"
        exit 1
    fi
}

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

    for name in mcp_server api frontend; do
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
    for port in 8001 8002 5173; do
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
    require_command python3 "未找到 python3，请先安装 Python 3"
    require_command npm "未找到 npm，请先安装 Node.js 和 npm"
    require_command lsof "未找到 lsof，start.sh 依赖它释放端口"
    require_command ss "未找到 ss，start.sh 依赖它检查端口状态"

    if [ ! -d "$FRONTEND_DIR" ]; then
        error "未找到前端目录: $FRONTEND_DIR"
        exit 1
    fi

    if [ ! -f "$FRONTEND_DIR/package.json" ]; then
        error "未找到 frontend/package.json，无法启动 Vue3 前端"
        exit 1
    fi

    # ---------- 1. MCP Server (port 8001) ----------
    if check_port 8001; then
        warn "端口 8001 已被占用，跳过 MCP Server 启动"
    else
        info "启动 MCP Server (port 8001) ..."
        nohup env PYTHONPATH="$PROJECT_DIR" python3 -m src.services.mcp_server \
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

    # ---------- 2. API Backend (port 8002) ----------
    if check_port 8002; then
        warn "端口 8002 已被占用，跳过 API Backend 启动"
    else
        info "启动 API Backend (port 8002) ..."
        nohup env PYTHONPATH="$PROJECT_DIR" python3 -m src.api.main \
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

    # ---------- 3. Vue3 Frontend (Vite, port 5173) ----------
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        warn "检测到 frontend/node_modules 不存在，正在安装前端依赖..."
        if [ -f "$FRONTEND_DIR/package-lock.json" ]; then
            (cd "$FRONTEND_DIR" && npm ci)
        else
            (cd "$FRONTEND_DIR" && npm install)
        fi
    fi

    if check_port 5173; then
        warn "端口 5173 已被占用，尝试释放..."
        kill -9 $(lsof -t -i :5173) 2>/dev/null || true
        sleep 2
        if check_port 5173; then
            error "无法释放端口 5173，请手动处理"
            exit 1
        fi
    fi

    info "启动 Vue3 Frontend (Vite, port 5173) ..."

    nohup bash -lc "cd \"$FRONTEND_DIR\" && npm run dev -- --host 0.0.0.0 --port 5173" \
        > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$PID_DIR/frontend.pid"
    sleep 3
    if kill -0 "$(cat "$PID_DIR/frontend.pid")" 2>/dev/null; then
        info "Vue3 Frontend 已启动 (PID $(cat "$PID_DIR/frontend.pid"))"
    else
        error "Vue3 Frontend 启动失败，请检查 $LOG_DIR/frontend.log"
        exit 1
    fi

    # ---------- 汇总 ----------
    echo ""
    echo "========================================="
    echo "  🚀 所有服务已启动"
    echo "========================================="
    echo ""
    echo "  MCP Server   : http://localhost:8001/mcp/"
    echo "  API Backend  : http://localhost:8002"
    echo "  Vue3 Frontend: http://localhost:5173"
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
    for name in mcp_server api frontend; do
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
        echo "  本地 9000 -> 远程 8002 (API Backend)"
        echo "  本地 9001 -> 远程 8001 (MCP Server)"
        echo "  本地 9002 -> 远程 5173 (Vue3 Frontend)"
        exit 1
    fi

    local api_port=$((local_base_port))
    local mcp_port=$((local_base_port + 1))
    local frontend_port=$((local_base_port + 2))

    echo "========================================="
    echo "  创建 SSH 隧道连接"
    echo "========================================="
    echo ""
    info "服务器: $server_host"
    echo ""
    echo "端口映射："
    echo "  本地 $api_port       -> 远程 8002 (API Backend)"
    echo "  本地 $mcp_port       -> 远程 8001 (MCP Server)"
    echo "  本地 $frontend_port  -> 远程 5173 (Vue3 Frontend)"
    echo ""
    warn "前端隧道注意事项："
    echo "  1. 浏览器访问: http://localhost:$frontend_port"
    echo "  2. 如遇页面异常，先检查 logs/frontend.log"
    echo ""
    info "正在建立隧道连接 (Ctrl+C 断开)..."
    echo ""

    ssh -N -L ${api_port}:localhost:8002 \
           -L ${mcp_port}:localhost:8001 \
           -L ${frontend_port}:localhost:5173 \
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
