#!/usr/bin/env bash
DEVICES=(mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7)
PORT=1
INTERVAL=5

bytes(){ cat /sys/class/infiniband/$1/ports/$PORT/counters/$2; }

while true; do
  declare -A TX1 RX1 TX2 RX2
  for D in "${DEVICES[@]}"; do
      RX1[$D]=$(bytes $D port_rcv_data)
      TX1[$D]=$(bytes $D port_xmit_data)
  done
  sleep $INTERVAL
  for D in "${DEVICES[@]}"; do
      RX2[$D]=$(bytes $D port_rcv_data)
      TX2[$D]=$(bytes $D port_xmit_data)
  done

  echo "$(date '+%F %T')"
  for D in "${DEVICES[@]}"; do
      RX_B=$(( ( ${RX2[$D]} - ${RX1[$D]} ) * 4 ))   # octets→bytes
      TX_B=$(( ( ${TX2[$D]} - ${TX1[$D]} ) * 4 ))
      RX_Mbps=$(echo "scale=1; $RX_B*8/1000000/$INTERVAL" | bc)
      TX_Mbps=$(echo "scale=1; $TX_B*8/1000000/$INTERVAL" | bc)
      printf "[%-6s] RX: %6s Mb/s | TX: %6s Mb/s\n" "$D" "$RX_Mbps" "$TX_Mbps"
  done
  echo
done
