import requests

payload = {
    "name": "Rotterdam Port",
    "bbox": [3.90, 51.82, 4.65, 52.08],
    "days_back": 90,
    "max_cloud": 40,
    "limit": 8,
}
res = requests.post("http://localhost:8787/api/analyze", json=payload, timeout=90)
print(res.status_code)
print(res.text[:2000])
