import json, statistics

with open("numbers.txt", "r") as file:
    numbers = [int(number) for number in file]

stats = {
    "mean": statistics.mean(numbers),
    "count": len(numbers),
    "max": max(numbers)
}

with open("outputs/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Output in output/stats.json")