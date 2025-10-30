"""Example plugin emulating Tessrax v15 contract."""

total = sum(payload.get("claims", []))
count = len(payload.get("claims", []))
result = {"total": total, "count": count}
