import requests
import sys

class VapdClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def detect(self, action_id: str, video_file: str, box: list) -> dict:
        """
        调用 vapd_detect 接口
        :param action_id: 任务ID
        :param video_file: 文件名 (会拼接成完整的URL)
        :param box: [x1,y1,x2,y2]
        :return: 接口返回的 JSON
        """
        video_url = (
            "https://inc-vsap-clos-shenzhen-xili1-oss.sit.sf-express.com/v1/"
            "AUTH_INC-VSAP-CLOS/test/fccv/vapd-recheck/"
            f"{video_file}"
            "?temp_url_sig=2a395d29897b0f43088e4493b12966d56658135d"
            "&temp_url_expires=1897385267"
            "&temp_url_prefix=fccv"
        )

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "action_id": action_id,
            "video_url": video_url,
            "box": box
        }

        resp = requests.post(f"{self.base_url}/vapd_detect", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def load_files(file_path: str):
    """
    从文件中读取文件名和box
    文件格式示例:
    021WG06867_21_172969_250910023227_00_00_1744.mp4  562,6,730,208
    """
    files = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            filename = parts[0]
            box = [int(x) for x in parts[1].split(",")]
            files.append((filename, box))
    return files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test.py <重复次数>")
        sys.exit(1)

    repeat = int(sys.argv[1])
    BASE_URL = "http://ai-gateway.sit.sf-express.com/openapi/modelserving/api/modelservice-10832"
    TOKEN = "eyJhbGciOiJIUzUxMiIsInppcCI6IkRFRiJ9.eNpUjE0LgkAQQP_LHMVh22ZXV2_hKQgLhO6jTrWRH-QWhPjfs2OnBw_em4Ff4Qb5DCP_CGoYpefRq25o5THJ8-37q_oXjaDeONqqKIIYps8UpCu5kzXflwWeq90Ji8OxgiWGe_CrdpQ4Y5kxbesGjUk01kYy5Dq7aKKmtUTrynOAXKfG2cykjpYvAAAA__8.1iQ1B6LXgo6hQOlUykE1IgCCd-shErIJy3e7W6FNUE74jSrYAA08q3eMZt0XXHvAuI9pb2A5lLf-vIumEHo-Pw"

    files = load_files("files.txt")
    client = VapdClient(BASE_URL, TOKEN)

    for filename, box in files:
        print(f"\n=== 开始测试文件 {filename} ===")
        for i in range(repeat):
            action_id = f"{filename}_run{i+1}"
            try:
                result = client.detect(action_id, filename, box)
                print(f"[{filename}] 第{i+1}次 -> {result.get('result')}")
            except Exception as e:
                print(f"[{filename}] 第{i+1}次 -> 调用失败: {e}")
