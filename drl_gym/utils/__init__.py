import csv


def get_experiment_csv_writer(file, player_count):
    fieldnames = ["game", "mean_time_per_game"]
    for i in range(player_count):
        fieldnames.append(f"mean_score_{i}")
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def write_experiment_row(writer, game, mean_scores, mean_time_per_game):
    row = {"game": game, "mean_time_per_game": mean_time_per_game}
    for i, mean_score in enumerate(mean_scores):
        row[f"mean_score_{i}"] = mean_score
    writer.writerow(row)
