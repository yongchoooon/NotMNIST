############################################################################
# This code is the code that takes the contents of the function called "slack_sender" from the KnockKnock library.
# I brought them to modify and use.

# Body Link : https://github.com/huggingface/knockknock
############################################################################

import datetime
import traceback
import json
import socket
import requests
import slack_config

class SlackSender:
  def __init__(self, title: str, state: str = None):
      self.webhook_url = slack_config.WEBHOOK_URL
      self.channel = slack_config.CHANNEL
      self.title = title
      self.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
      self.state = state

      self.start_time = datetime.datetime.now()
      self.host_name = socket.gethostname()

      self.slack_sender(state = self.state)


  def slack_sender(self, state: str, plt_dir: str = None, epoch: int = None, value = None):
      dump = {
          "username": "Knock Knock",
          "channel": self.channel
      }

      start_time = self.start_time
      host_name = self.host_name

      contents = []

      try: 
        if state == "start":
          contents = (["Your training has started 🎬",
                          "Machine name: %s" % host_name,
                          "Title: %s" % self.title,
                          "Starting date: %s" % start_time.strftime(self.DATE_FORMAT)])
          dump["text"] = "\n".join(contents)
          dump["icon_emoji"] = ":clapper:"
        
        elif state == "training":
          end_time = datetime.datetime.now()
          elapsed_time = end_time - start_time
          contents = (["Your training is going well 🔥",
                          "Machine name: %s" % host_name,
                          "Title: %s" % self.title,
                          "Starting date: %s" % start_time.strftime(self.DATE_FORMAT),
                          "End date: %s" % end_time.strftime(self.DATE_FORMAT),
                          "Training duration: %s" % str(elapsed_time),
                          "============================="])
          
          try:
            str_value = str(value)
            contents.append(str_value)
            contents.append("=============================")
          except:
            contents.append("Title returned value: %s"% "ERROR - Couldn't str the returned value.")

          dump["text"] = "\n".join(contents)
          dump["username"] = "Knock Knock - Training"
          dump["icon_emoji"] = ":fire:"

          self.slack_plt_image_sender(plt_dir, epoch)

        elif state == "end":
          end_time = datetime.datetime.now()
          elapsed_time = end_time - start_time
          contents = (["Your training is complete 🎉",
                          "Machine name: %s" % host_name,
                          "Title: %s" % self.title,
                          "Starting date: %s" % start_time.strftime(self.DATE_FORMAT),
                          "End date: %s" % end_time.strftime(self.DATE_FORMAT),
                          "Training duration: %s" % str(elapsed_time)])

          dump["text"] = "\n".join(contents)
          dump["username"] = "Knock Knock - End"
          dump["icon_emoji"] = ":tada:"

        requests.post(self.webhook_url, json.dumps(dump))

      except Exception as ex:
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        contents = (["Your training has crashed ☠️",
                        'Machine name: %s' % host_name,
                        'Title: %s' % self.title,
                        'Starting date: %s' % start_time.strftime(self.DATE_FORMAT),
                        'Crash date: %s' % end_time.strftime(self.DATE_FORMAT),
                        'Crashed training duration: %s\n\n' % str(elapsed_time),
                        "Here's the error:",
                        '%s\n\n' % ex,
                        "Traceback:",
                        '%s' % traceback.format_exc()])
        dump['text'] = '\n'.join(contents)
        dump['icon_emoji'] = ':skull_and_crossbones:'
        requests.post(self.webhook_url, json.dumps(dump))
        raise ex

  def slack_plt_image_sender(self, plt_dir, epoch):
      with open(plt_dir / f"plt-epoch{str(epoch)}.png", "rb") as f: ## 이미지 보내기 
          header = {
            'Content-type': 'application/x-www-form-urlencoded; charset=utf-8',
            'Authorization': "Bearer " + slack_config.TOKEN}
          attachments = {
            "user": slack_config.USER_ID,
            "channels": slack_config.CHANNEL_ID,
            "title": self.title + " - Epoch: %s" % epoch,
            "content": f.read()
          }
      requests.post('https://slack.com/api/files.upload', headers = header, data = attachments)