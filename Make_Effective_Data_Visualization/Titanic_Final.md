
# Titanic_Data_Visualization

## Summary: choice of dataset and data visualization goal

The dataset I prefered to make visualization is Titanic dataset which contains a sample of demographic data and some information regarding 891 passengers. Although there was some sort of fate for some passengers in surviving the sinking, based on investigation of this dataset, some groups of people were more likely to survive than others; such as women, children, and people in the upper-class. I intend to visualize the likelihood of survival based on passengers age. For plotting I will use dimple.js library. 

## Design 

In order to visualize the likelihood of survival/death for Titanic passengers, first I grouped my dataset in three different age groups; "younger age (below 20 years)", "middle age (20 to 50 years)" and " Old age (Over 50 years)". My goal is to show relative differences within each category, so I used a stacked percentage column chart. I took a broader age classification becasue too many categories per group add visual noise, making it hard to see the patterns in the data.
At the begining I tried to take the total number of survival/deaths in each category, but I discovered this will not serve for my purpose so I dropped it and decided to take the percentage.After I gathred feedbacks, I ended up focussing on survival rate instead of showing survival/deaths rates as people will clearly understand the rate of death by subtracting from 100%.

## Takeaway message for readers

Based on the Titanic dataset exploration, it was clear that some groups of people were more likely to survive than others; such as women, children, and people in the upper-class. Visualization of the same dataset (from age of passegers point of view) also explains the same story; the rate of survival diminished with the increment of age (younger passengers had high likelihood of survival compared to middle aged and old aged passengers). 

## Feedback

The feedbacks that I collected after I showed my first sketch to three different people are summarized below: 

•	Your chart is simple and clear. I like the color and it matches with the content of the message. I guess you are trying to tell us that many young people survived and a lot of old people died. I mean, I understand that there is high survival for young people compared to middle age and old age people. This also means there is high death for old people compared to middle and old ages. It is pretty clear. One thing I noticed, people do not interpret proportion easily so why don’t use rate instead.


•	Nice chart. Based on the title I guess you are focusing on survival proportion, but I feel like you can change colors so that people will not think your focus is on death proportion. By the way, use percentage instead of proportion for clear understanding. On mouth hovering one can understand the exact value, but I suggest you display the values on the bar. One more thing, don’t you think it is a good idea to mention how old are middle aged, or old aged. For people not to interpret their own way, give them the exact amount of years you mean by each age classification. Finally, let me tell you what I understood from this graph, your graph is showing that survival is increasing as age increases and in the contrary deaths are high for older people. 


•	I find your graph easy to follow, however I would like to know what do you mean by younger age, how old are they? I feel it is important to make it clear here. You know what you mean by that, but readers and viewers may not know it. So I suggest you include age lebels and it will be more clear. Again, I can understand what you mean by proportion, but I afraid other people may not. It is good to user percentages I feel. In general, I noticed that survival is going down across the different age classifications. It also shows that death is going up. May be since your title is about survival, you can play around with color to give more emphasis on survival. Well, it is obvious that survival and death are two faces of a coin. I feel if you just take into consideration the two points I mentioned, your chart will be pretty cool. 


In my second sketch, I incorporated all the comments given 

•	I took percent of survival/death instead of proportion 

•	I added lebels to Age in brackets; younger age “below 20 years”, middle age “20 to 50 years” and old age “50 and above”.

•	I included the values (percent of survival/death) for each group.

My final sketch included comments given by my reviewer.

## References

http://www.scribblelive.com/blog/2012/08/27/how-groups-stack-up-when-to-use-grouped-vs-stacked-column-charts/

http://dimplejs.org/examples_viewer.html?id=bars_vertical_grouped_stacked

http://blog.visual.ly/creating-animations-and-transitions-with-d3-js/

https://www.dashingd3js.com/lessons/basic-chart-grouped-bar-chart

http://dimplejs.org/advanced_examples_viewer.html?id=advanced_bar_labels

http://colorbrewer2.org/

https://resources.oncourse.iu.edu/access/content/user/rreagan/Filemanager_Public_Files/meaningofcolors.htm
