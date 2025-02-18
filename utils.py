from datetime import datetime
import os
import requests

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neural_dnf.neural_dnf import BaseNeuralDNF

################################################################################
#                                   Plotting                                   #
################################################################################


def generate_weight_histogram(
    model: BaseNeuralDNF, num_bins: int = 20
) -> tuple[Figure, Figure]:
    conj_w = model.conjunctions.weights.data.flatten().detach().cpu().numpy()
    disj_w = model.disjunctions.weights.data.flatten().detach().cpu().numpy()

    f1 = plt.figure(figsize=(20, 15))
    plt.title("Conjunction weight distribution")
    arr = plt.hist(conj_w, bins=num_bins)
    for i in range(num_bins):
        plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))  # type: ignore

    f2 = plt.figure(figsize=(20, 15))
    plt.title("Disjunction weight distribution")
    arr = plt.hist(disj_w, bins=num_bins)
    for i in range(num_bins):
        plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))  # type: ignore

    return f1, f2


################################################################################
#                              Webhook utils                                   #
################################################################################


def post_to_discord_webhook(
    webhook_url: str,
    experiment_name: str,
    message_body: str,
    errored: bool,
    keyboard_interrupt: bool | None = None,
) -> None:
    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    nodename = os.uname().nodename
    if keyboard_interrupt:
        message_head = (
            f"[{dt}]\n"
            f"Experiment {experiment_name} on hine {nodename} "
            f"INTERRUPTED!!\n"
        )
    else:
        message_head = (
            f"[{dt}]\n"
            f"Experiment {experiment_name} on hine {nodename} "
            f"{'ERRORED' if errored else 'FINISHED'}!!\n"
        )

    requests.post(webhook_url, json={"content": message_head + message_body})
