export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm_api

MODEL_API="http://0.0.0.0:8000"
SERVE_MODEL_NAME="SERVE_MODEL_NAME"

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

wait_api_ready() {
    local api_base=$1
    echo "Waiting for vLLM serve API at $api_base..."
    while true; do
        if curl -s -o /dev/null -w "%{http_code}" "$api_base/health" | grep -q "200"; then
            break
        else
            sleep 5
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

wait_api_ready $MODEL_API

echo "vLLM serve API is ready"

nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py --backend $BACKEND --n-logprobs 256 --api-base $MODEL_API --serve-model $SERVE_MODEL_NAME &> worker.log &
echo "start teacher worker"

echo "teacher server is ready"