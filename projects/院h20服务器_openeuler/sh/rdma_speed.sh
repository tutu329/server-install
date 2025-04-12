#!/bin/bash

# 设置要监控的两个网卡
IFACES=("ens19f0np0" "ens19f1np1")

# 间隔时间（秒）
INTERVAL=5

echo "Monitoring interfaces: ${IFACES[*]} every $INTERVAL seconds..."
echo "Press Ctrl+C to stop."
echo ""

# 无限循环
while true; do
    declare -A RX1 TX1 RX2 TX2

    for IFACE in "${IFACES[@]}"; do
        RX1[$IFACE]=$(ethtool -S $IFACE | grep rx_bytes_phy | awk '{print $2}')
        TX1[$IFACE]=$(ethtool -S $IFACE | grep tx_bytes_phy | awk '{print $2}')
    done

    TIME1=$(date +%s)
    sleep $INTERVAL
    TIME2=$(date +%s)

    for IFACE in "${IFACES[@]}"; do
        RX2[$IFACE]=$(ethtool -S $IFACE | grep rx_bytes_phy | awk '{print $2}')
        TX2[$IFACE]=$(ethtool -S $IFACE | grep tx_bytes_phy | awk '{print $2}')
    done

    TIME_DIFF=$((TIME2 - TIME1))

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    for IFACE in "${IFACES[@]}"; do
        RX_DIFF=$(( ${RX2[$IFACE]} - ${RX1[$IFACE]} ))
        TX_DIFF=$(( ${TX2[$IFACE]} - ${TX1[$IFACE]} ))

        RX_Mbps=$(echo "scale=2; $RX_DIFF * 8 / 1000000 / $TIME_DIFF" | bc)
        TX_Mbps=$(echo "scale=2; $TX_DIFF * 8 / 1000000 / $TIME_DIFF" | bc)

        echo "[$IFACE] RX: ${RX_Mbps} Mbps | TX: ${TX_Mbps} Mbps"
    done

    echo ""
done
