def recommend(data):

    suggestions = []

    if data["study_hours"] < 3:
        suggestions.append("Increase study hours")

    if data["attendance"] < 75:
        suggestions.append("Improve attendance")

    if data["social_media_hours"] > 4:
        suggestions.append("Reduce social media usage")

    if data["sleep_hours"] < 6:
        suggestions.append("Maintain proper sleep")

    return suggestions