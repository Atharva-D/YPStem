# YPStem

## Inspiration:
In a world rapidly moving towards inclusivity in all domains, features allowing accessibility to the deaf and hard of hearing community gain a lot of importance. When the features are related to emergency services, the services are requirements, not a luxury.
Across the globe, emergency services rely on a telephone/mobile number/hotline, to be contacted by the affected in difficult situations. People using SL(Sign Language) as their means of communication lose their right to these fundamental services. The inspiration for this project,thus, comes from the necessity of providing access to emergency services during times of crisis.

## What it does
The website, named "", interprets ASL components for "accident","ambulance","breathe","emergency","fire" and "help" which are basic signs which can be instrumental in any catastrophic situations. On detecting a specific sign, a message is sent to the emergency service contact number, including the name of the emergency faced. Relying on the Text-to-911 functionality,"this website" allows emergency services to be requested, with just the ASL signs shown by the affected.

## How we built it
For building "", we have created an LSTM(Long short-term memory) machine learning model, utilizing a custom dataset for the ASL signs "accident", "ambulance", "breathe", "emergency","fire" and "help". The model yielded "%" accuracy with MediaPipe solutions for detecting hand signs. Deployed on the "" website using Flask......

## Challenges we ran into
The main challenges we faced include a lack of a ready dataset for emergency signs, the difficulty in decoding signs which were not just static, but involving continuous movement, and the obstacle in messaging the emergency service contact number directly from the website.

## Accomplishments that we're proud of
The ML model used can efficiently detect the ASl signs, with an accuracy of "%". Just showing the sign, a person can directly contact the emergency sevice in a matter of second..

## What we learned


## What's next for Untitled
The next step for "" has to be the inclusion of many more dynamic ASL signs, making "" a leading solution for accessibility issues our fellow humans face.
