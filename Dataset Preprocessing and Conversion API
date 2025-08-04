#datasets Tramslation into english resulted (mix of multilinhual tweets due to incomplete translation of individual tweet text) 
import csv
TwitterId = []
Content = []
Tweets = []
GPS = []
Localization = []
Time = []
Classification = []
Generated_Useful_Info_at_BOD = []

with open('FinalDataset/FinalDataset3.csv', 'w', encoding='Latin-1', newline='') as csvfile:
       writer = csv.writer(csvfile)
       writer.writerow(['TwitterId', 'Content', 'Tweets', 'GPS', 'Localization', 'Time', 'Classification'])
       for i in range(len(df['Tweets'])):
           if (df['Tweets'][i]!='Loading...'):
               {
                   writer.writerow([df['TwitterId'][i] , df['Content'][i], df['Tweets'][i], df['GPS'][i], df['Localization'][i], df['Time'][i], df['Classification'][i]])
               }
