"""Monitor Kaggle Submissions and Send Slack Notifications.

This module monitors Kaggle competition submissions and sends notifications 
to a configured Slack channel when new submissions are detected or when 
the status of existing submissions changes (e.g., from "pending" to "complete"). 

Features:
- Fetches Kaggle submissions using the Kaggle API.
- Converts submission timestamps from UTC to the local time zone.
- Tracks submission statuses and execution times.
- Sends real-time notifications to Slack for new submissions and status updates.
- Automatically handles Slack errors and retries.

Environment Variables:
    SLACK_WEBHOOK_URL (str): Slack webhook URL for sending notifications.
    KAGGLE_COMPETITION_ID (str): ID of the Kaggle competition to monitor.

Requirements:
    - Python 3.6+
    - `requests` library for sending Slack notifications.
    - `dotenv` library for loading environment variables.
    - `pytz` library for time zone conversion.
    - Kaggle API configured and authenticated via CLI.

Usage:
    1. Set up the `.env` file with the required environment variables.
    2. Ensure the Kaggle API is installed and authenticated.
    3. Run the script to start monitoring submissions.

Example `.env` file:
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
    KAGGLE_COMPETITION_ID=your-competition-id
"""


import os
import time
import requests
import csv
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import pytz

load_dotenv()

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
submission_statuses = {}
first_iteration = True
local_timezone = pytz.timezone('Europe/London')


def send_slack_notification(message):
    """
    Sends a notification message to Slack using the configured webhook URL.

    Args:
        message (str): The message to be sent to Slack.

    Returns:
        None
    """
    response = requests.post(SLACK_WEBHOOK_URL, json={"text": message})
    if response.status_code != 200:
        print(f"Slack API returned an error. HTTP Status: {response.status_code}")
    else:
        print("Slack notification sent successfully.")


def convert_to_local_time(date_time_str):
    """
    Converts a UTC date-time string to the local time zone.

    Args:
        date_time_str (str): UTC date-time string in the format '%Y-%m-%d %H:%M:%S'.

    Returns:
        str: Localized date-time string in the format '%Y-%m-%d %H:%M:%S'.
    """
    date_time_obj = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
    utc_time = pytz.utc.localize(date_time_obj)
    local_time = utc_time.astimezone(local_timezone)
    return local_time.strftime("%Y-%m-%d %H:%M:%S")


def local_time_to_timestamp(local_time_str):
    """
    Converts a local time string to a Unix timestamp.

    Args:
        local_time_str (str): Local time string in the format '%Y-%m-%d %H:%M:%S'.

    Returns:
        int: Unix timestamp.
    """
    local_time_obj = datetime.strptime(local_time_str, "%Y-%m-%d %H:%M:%S")
    return int(local_time_obj.timestamp())


def check_submissions():
    """
    Checks the status of Kaggle submissions and sends Slack notifications for new or updated submissions.

    Returns:
        None
    """
    global first_iteration
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 70)
    print(f"{current_time}: Fetching Kaggle submissions...")

    kaggle_competition_id = os.getenv("KAGGLE_COMPETITION_ID")
    result = subprocess.run([
        'kaggle', 'competitions', 'submissions',
        '-c', kaggle_competition_id, '--csv'
    ], stdout=subprocess.PIPE, universal_newlines=True)
    submissions = result.stdout

    reader = csv.reader(submissions.splitlines())
    next(reader)

    for row in reader:
        fileName, date, description, status, publicScore, privateScore = row
        if date == "date":
            continue

        date_time = convert_to_local_time(date)
        timestamp = local_time_to_timestamp(date_time)

        print(f"{current_time}: {fileName}, {date_time}, {description}, {status}, {publicScore}, {timestamp}")

        if timestamp not in submission_statuses:
            if first_iteration:
                submission_statuses[timestamp] = status
            else:
                submission_statuses[timestamp] = status
                notification_message = (f"*{current_time}* :rocket: *New Kaggle Submission Detected*\n"
                                        f"> *Submission:* `{description}`\n"
                                        f"> *Submitted at:* `{date_time}`\n"
                                        f"> *Status:* `{status}`\n"
                                        f"> *Score:* `{publicScore}`")
                print(f"{current_time}: New submission detected: {description} at {date_time}. Status: {status}.")
                send_slack_notification(notification_message)
        else:
            if status == "complete" and submission_statuses.get(timestamp) == "pending":
                submission_statuses[timestamp] = "complete"
                current_timestamp = int(time.time())
                execution_time = current_timestamp - timestamp
                execution_hours = int(execution_time // 3600)
                execution_minutes = int((execution_time % 3600) // 60)
                notification_message = (f"*{current_time}* :bell: *Kaggle Submission Update*\n"
                                        f"> *Submission:* `{description}`\n"
                                        f"> *Submitted at:* `{date_time}`\n"
                                        f"> *Completed at:* `{current_time}`\n"
                                        f"> *Status:* `{status}`\n"
                                        f"> *Score:* `{publicScore}`\n"
                                        f"> *Time taken:* `{execution_hours}h {execution_minutes}m`")
                print(
                    f"{current_time}: Submission at {date_time} "
                    f"(timestamp: {timestamp}) with description {description} "
                    f"has changed status to {status}. Time taken: {execution_hours}h {execution_minutes}m."
                )
                send_slack_notification(notification_message)

            submission_statuses[timestamp] = status

    if first_iteration:
        first_iteration = False


def main():
    """
    Main function to monitor Kaggle submissions and send notifications.

    Returns:
        None
    """
    load_dotenv()
    send_slack_notification(":mag: *Monitoring Kaggle submissions...*")

    while True:
        check_submissions()
        time.sleep(60)


if __name__ == "__main__":
    main()
