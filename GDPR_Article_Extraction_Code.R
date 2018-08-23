#Imports
library(rvest)
 

#Variables
Base_URL <- "https://gdpr-info.eu/chapter-3/"
#please chnage the file path
output_file_path <- "D:\\DATA ANALYTICS\\ASSIGNMENT\\2nd sem\\adm\\project\\deliverables\\extracted_articles\\"

#logic

#Extraction of all article links
raw_html <- read_html(Base_URL)
links_details <- raw_html %>% html_nodes('.sub-menu li a')
links <- links_details %>% html_attr('href')

#For each article link get the article content and save it in text file
for(link in links){
  print(paste("Processing page ", link, "....."))
  page_content <- read_html(link)
  page_id <- page_content %>% html_nodes('.entry-header .dsgvo-number')%>%html_text()
  page_title <- page_content %>% html_nodes('.entry-header .dsgvo-title')%>%html_text()
  page_data <- page_content %>% html_nodes('.entry-content ol li')%>%html_text()
  fileConn<-file(paste(output_file_path,trimws(page_id),'.','tsv',sep=""))
  writeLines(page_title, fileConn) # save first line as title
  writeLines(page_data, fileConn)  # append contet to text file
  close(fileConn)
}

