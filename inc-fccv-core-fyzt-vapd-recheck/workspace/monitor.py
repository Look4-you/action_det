import os
import threading
import time
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

class RequestMonitor:
    """
    单指标并发监控器：
      - 每秒采样一次当前值
      - 每 monitor_interval 秒输出 busy ratio + 详细样本
      - 每 summary_interval_minutes 分钟输出 busy ratio（仅比例）
    """

    def __init__(
        self,
        name: str,
        monitor_interval: int = None,
        summary_interval_minutes: int = None,
    ):
        self.name = name
        self._current = 0
        self._detail_counts = []
        self._summary_counts = []
        self._lock = threading.Lock()

        # 从环境变量读取或使用参数
        self.monitor_interval = (
            monitor_interval
            if monitor_interval is not None
            else int(os.getenv("MONITOR_INTERVAL", "60"))
        )
        summary_min = (
            summary_interval_minutes
            if summary_interval_minutes is not None
            else int(os.getenv("SUMMARY_INTERVAL_MINUTES", "30"))
        )
        self.summary_interval = summary_min * 60

        # 启动后台线程
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def update(self, count: int):
        """直接将当前并发设置为 count"""
        with self._lock:
            self._current = count

    def inc(self, delta: int = 1):
        """并发 +delta"""
        with self._lock:
            self._current += delta

    def dec(self, delta: int = 1):
        """并发 -delta（不低于0）"""
        with self._lock:
            self._current = max(0, self._current - delta)

    def _run(self):
        next_detail = time.time() + self.monitor_interval
        next_summary = time.time() + self.summary_interval

        while True:
            time.sleep(1)
            with self._lock:
                cnt = self._current

            # 记录到两个窗口
            self._detail_counts.append(cnt)
            self._summary_counts.append(cnt)

            now = time.time()

            # 细节报告
            if now >= next_detail:
                busy = sum(1 for x in self._detail_counts if x > 0)
                ratio = busy / len(self._detail_counts) if self._detail_counts else 0
                logging.info(
                    f"[{self.name}] detail busy ratio: {ratio:.2%}, samples: {self._detail_counts}"
                )
                self._detail_counts.clear()
                next_detail += self.monitor_interval

            # 汇总报告
            if now >= next_summary:
                busy = sum(1 for x in self._summary_counts if x > 0)
                ratio = busy / len(self._summary_counts) if self._summary_counts else 0
                logging.info(f"[{self.name}] summary busy ratio: {ratio:.2%}")
                self._summary_counts.clear()
                next_summary += self.summary_interval
