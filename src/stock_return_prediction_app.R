library(shiny)
library(ggplot2)
library(dplyr)
library(plotly)
library(reshape2)
library(readr)

# Get data
start_date <- as.Date("1985-01-01")
end_date <- as.Date("2018-12-31")

Group_A_X <- read.csv("mef_daily_data_processed.csv")
Group_A_Y <- read.csv("mai_daily_data_processed.csv")
Group_A_Z <- read.csv("mkt_daily_data_processed.csv")

Group_B_X <- read.csv("mef_monthly_data_processed.csv")
Group_B_Y <- read.csv("mai_monthly_data_processed.csv")
Group_B_Z <- read.csv("mkt_monthly_data_processed.csv")

Group_C_X <- read.csv("mef_quarterly_data_processed.csv")
Group_C_Y <- read.csv("mai_quarterly_data_processed.csv")
Group_C_Z <- read.csv("mkt_quarterly_data_processed.csv")

# Vector of linetypes
linetypes <- c("solid", "dashed", "dotted", "dotdash", "longdash", "twodash")

# Define UI
ui <- fluidPage(
  titlePanel("Data Visualization: MEF & MAI Data vs. MKT Data"),
  sidebarLayout(
    sidebarPanel(
      selectInput("group_select", "Select Data Frequency:", choices = c("Daily", "Monthly", "Quarterly")),
      checkboxGroupInput("feature_X_select", "Select Variables from MEF Data:", choices = colnames(Group_A_X)[2:15], selected = colnames(Group_A_X)[2:3], inline = TRUE),
      checkboxGroupInput("feature_Y_select", "Select Variables from MAI Data:", choices = colnames(Group_A_Y)[2:9], selected = colnames(Group_A_Y)[2:3], inline = TRUE),
      selectInput("feature_Z_select", "Select Index from MKT Data:", choices = colnames(Group_A_Z)[2:2], selected = colnames(Group_A_Z)[2]),
      dateRangeInput("date_range", "Select Date Range:", start = start_date, end = end_date)
    ),
    mainPanel(
      plotOutput("lineplot"),
      plotOutput("heatmap")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Updated selected_group function with date range filter
  filtered_data <- reactive({
    switch(input$group_select,
           "Daily" = {
             data_X <- Group_A_X[Group_A_X$date >= input$date_range[1] & Group_A_X$date <= input$date_range[2], c("date", input$feature_X_select)]
             data_Y <- Group_A_Y[Group_A_Y$date >= input$date_range[1] & Group_A_Y$date <= input$date_range[2], c("date", input$feature_Y_select)]
             data_Z <- Group_A_Z[Group_A_Z$date >= input$date_range[1] & Group_A_Z$date <= input$date_range[2], c("date", input$feature_Z_select)]
           },
           "Monthly" = {
             data_X <- Group_B_X[Group_B_X$date >= input$date_range[1] & Group_B_X$date <= input$date_range[2], c("date", input$feature_X_select)]
             data_Y <- Group_B_Y[Group_B_Y$date >= input$date_range[1] & Group_B_Y$date <= input$date_range[2], c("date", input$feature_Y_select)]
             data_Z <- Group_B_Z[Group_B_Z$date >= input$date_range[1] & Group_B_Z$date <= input$date_range[2], c("date", input$feature_Z_select)]
           },
           "Quarterly" = {
             data_X <- Group_C_X[Group_C_X$date >= input$date_range[1] & Group_C_X$date <= input$date_range[2], c("date", input$feature_X_select)]
             data_Y <- Group_C_Y[Group_C_Y$date >= input$date_range[1] & Group_C_Y$date <= input$date_range[2], c("date", input$feature_Y_select)]
             data_Z <- Group_C_Z[Group_C_Z$date >= input$date_range[1] & Group_C_Z$date <= input$date_range[2], c("date", input$feature_Z_select)]
           })
    
    # Normalize the data based on mean and standard deviation
    normalize_data <- function(data) {
      numeric_cols <- sapply(data, is.numeric)
      data[numeric_cols] <- lapply(data[numeric_cols], function(col) {
        (col - mean(col, na.rm = TRUE)) / sd(col, na.rm = TRUE)
      })
      data
    }
    
    list(normalize_data(data_X), normalize_data(data_Y), normalize_data(data_Z))
  })
  
  # Dynamically generate line styles for selected features
  linestyles_X <- reactive({
    rep(linetypes, length.out = length(input$feature_X_select))
  })
  
  linestyles_Y <- reactive({
    rep(linetypes, length.out = length(input$feature_Y_select))
  })
  
  # Generate line plot based on user input for X, Y, and Z
  output$lineplot <- renderPlot({
    # Store the selected data in variables
    data_X <- filtered_data()[[1]]
    data_Y <- filtered_data()[[2]]
    data_Z <- filtered_data()[[3]]
    
    selected_features_X <- input$feature_X_select
    selected_features_Y <- input$feature_Y_select
    selected_feature_Z <- input$feature_Z_select
    
    ggplot() +
      lapply(seq_along(input$feature_X_select), function(i) {
        feature <- selected_features_X[i]
        linetype <- linestyles_X()[i]
        geom_line(data = data_X, aes(x = seq_along(data_X[[feature]]), y = data_X[[feature]], linetype = linetype),
                  color = "blue", size = 1)
      }) +
      lapply(seq_along(input$feature_Y_select), function(i) {
        feature <- selected_features_Y[i]
        linetype <- linestyles_Y()[i]
        geom_line(data = data_Y, aes(x = seq_along(data_Y[[feature]]), y = data_Y[[feature]], linetype = linetype),
                  color = "green", size = 1)
      }) +
      geom_line(data = data_Z, aes(x = seq_along(data_Z[[selected_feature_Z]]), y = data_Z[[selected_feature_Z]]),
                color = "red", size = 1, linetype = "solid") +
      labs(title = "Normalized Line Plot",
           y = "Normalized Value") +
      scale_linetype_manual(
        values = c(
          linestyles_X(),
          linestyles_Y(),
          rep("solid", length(input$feature_Z_select))
        )
      ) +
      scale_color_manual(
        values = c(
          rep("blue", length(input$feature_X_select)),
          rep("green", length(input$feature_Y_select)),
          rep("red", length(input$feature_Z_select))
        ),
        breaks = c("Data X", "Data Y", "Data Z")
      ) +
      theme(legend.position = "none")
  })

  
  # Generate heatmap for correlation using ggplot2
  output$heatmap <- renderPlot({
    # Store the selected data in variables
    data_X <- filtered_data()[[1]]
    data_Y <- filtered_data()[[2]]
    data_Z <- filtered_data()[[3]]
    
    # Combine data into a single data frame
    if (length(input$feature_X_select) == 0 && length(input$feature_Y_select) == 0) {
      data_combined <- cbind(data_Z)
    }
    if (length(input$feature_X_select) == 0 && length(input$feature_Y_select) > 0) {
      data_combined <- cbind(data_Y, data_Z)
    }
    if (length(input$feature_X_select) > 0 && length(input$feature_Y_select) == 0) {
      data_combined <- cbind(data_X, data_Z)
    }
    if (length(input$feature_X_select) > 0 && length(input$feature_Y_select) > 0) {
      data_combined <- cbind(data_X, data_Y, data_Z)
    }
    
    # Remove the 'date' column
    data_combined <- data_combined[, !colnames(data_combined) %in% "date"]
    
    # Calculate the correlation matrix
    correlation_matrix <- cor(data_combined)
    
    # Prepare data for ggplot2 heatmap
    heatmap_data <- melt(correlation_matrix)
    
    # Plot the heatmap with ggplot2
    ggplot(data = heatmap_data, aes(Var1, Var2, fill = value)) +
      geom_tile() +
      coord_fixed(ratio = 1) +  # Ensure equal aspect ratio
      theme_minimal() +
      labs(title = "Correlation Heatmap",
           x = "",
           y = "",
           fill = "Correlation") +
      scale_fill_gradient2(low = "#1f78b4", mid = "white", high = "#e31a1c", midpoint = 0)
  })
  
}

# Run the application
shinyApp(ui, server)
