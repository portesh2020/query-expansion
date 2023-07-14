from googleapiclient.discovery import build


class GoogleApiClient:
  def __init__(self, developerKey, searchEngineId):
    self.service = build("customsearch", "v1", developerKey=developerKey)
    self.searchEngineId = searchEngineId


  def getQueryResults(self, query):
    return self.service.cse().list(q=query, cx=self.searchEngineId).execute()["items"]