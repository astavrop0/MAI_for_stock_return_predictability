library(shiny)
library(ggplot2)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Generate random data for Group A
start_date <- as.Date("1985-01-01")
end_date <- as.Date("2020-01-10")
dates <- seq(start_date, end_date, by = "days")

Group_A_X <- data.frame(
  date = dates,
  feature_1X = rnorm(length(dates)),
  feature_2X = rnorm(length(dates)),
  feature_3X = rnorm(length(dates))
)

Group_A_Y <- data.frame(
  date = dates,
  feature_1Y = rnorm(length(dates)),
  feature_2Y = rnorm(length(dates)),
  feature_3Y = rnorm(length(dates))
)

Group_A_Z <- data.frame(
  date = dates,
  feature_1Z = rnorm(length(dates)),
  feature_2Z = rnorm(length(dates)),
  feature_3Z = rnorm(length(dates))
)

# Generate random data for Group B
Group_B_X <- data.frame(
  date = dates,
  feature_1X = rnorm(length(dates)),
  feature_2X = rnorm(length(dates)),
  feature_3X = rnorm(length(dates))
)

Group_B_Y <- data.frame(
  date = dates,
  feature_1Y = rnorm(length(dates)),
  feature_2Y = rnorm(length(dates)),
  feature_3Y = rnorm(length(dates))
)

Group_B_Z <- data.frame(
  date = dates,
  feature_1Z = rnorm(length(dates)),
  feature_2Z = rnorm(length(dates)),
  feature_3Z = rnorm(length(dates))
)

# Generate random data for Group B
Group_C_X <- data.frame(
  date = dates,
  feature_1X = rnorm(length(dates)),
  feature_2X = rnorm(length(dates)),
  feature_3X = rnorm(length(dates))
)

Group_C_Y <- data.frame(
  date = dates,
  feature_1Y = rnorm(length(dates)),
  feature_2Y = rnorm(length(dates)),
  feature_3Y = rnorm(length(dates))
)

Group_C_Z <- data.frame(
  date = dates,
  feature_1Z = rnorm(length(dates)),
  feature_2Z = rnorm(length(dates)),
  feature_3Z = rnorm(length(dates))
)


# Define UI
ui <- fluidPage(
  titlePanel("Random Data Visualization"),
  sidebarLayout(
    sidebarPanel(
      dateRangeInput("date_range", "Select Date Range:",
                     start = start_date,
                     end = end_date),
      selectInput("group_select", "Select Group:", choices = c("Group A", "Group B" , "Group C")),
      checkboxGroupInput("feature_X", "Select Features from Data X:", choices = colnames(Group_A_X)[2:4], selected = c("feature_1X")),
      checkboxGroupInput("feature_Y", "Select Features from Data Y:", choices = colnames(Group_A_Y)[2:4], selected = c("feature_1Y")),
      selectInput("feature_Z", "Select Feature from Data Z:", choices = colnames(Group_A_Z)[2:4], selected = "feature_1Z")
    ),
    mainPanel(
      plotOutput("lineplot"),
      plotOutput("heatmap")
    )
  )
)

# Define server logic
server <- function(input, output) {
  # Reactive expression to filter data based on user input for X, Y, and Z
  filtered_data_X <- reactive({
    data <- switch(input$group_select,
                   "Group A" = Group_A_X,
                   "Group B" = Group_B_X,
                   "Group C" = Group_C_X)
    data <- data[data$date >= input$date_range[1] & data$date <= input$date_range[2], ]
    data
  })
  
  filtered_data_Y <- reactive({
    data <- switch(input$group_select,
                   "Group A" = Group_A_Y,
                   "Group B" = Group_B_Y,
                   "Group C" = Group_C_Y)
    data <- data[data$date >= input$date_range[1] & data$date <= input$date_range[2], ]
    data
  })
  
  filtered_data_Z <- reactive({
    data <- switch(input$group_select,
                   "Group A" = Group_A_Z,
                   "Group B" = Group_B_Z,
                   "Group C" = Group_C_Z)
    data <- data[data$date >= input$date_range[1] & data$date <= input$date_range[2], ]
    data
  })
  
  # Generate line plot based on user input for X, Y, and Z
  output$lineplot <- renderPlot({
    # Store the selected data in variables
    data_X <- filtered_data_X()
    data_Y <- filtered_data_Y()
    data_Z <- filtered_data_Z()
    
    ggplot() +
      lapply(input$feature_X, function(feature) {
        geom_line(data = data_X, aes(x = date, y = data_X[[feature]]),
                  color = "blue",
                  linetype = sample(c("solid", "dashed", "dotted"), 1),
                  size = sample(c(1, 1.5, 2), 1))
      }) +
      lapply(input$feature_Y, function(feature) {
        geom_line(data = data_Y, aes(x = date, y = data_Y[[feature]]),
                  color = "green",
                  linetype = sample(c("solid", "dashed", "dotted"), 1),
                  size = sample(c(1, 1.5, 2), 1))
      }) +
      lapply(input$feature_Z, function(feature) {
        geom_line(data = data_Z, aes(x = date, y = data_Z[[feature]]),
                  color = "red",
                  linetype = "solid",
                  size = 1)
      }) +
      labs(title = "Line Plot",
           x = "Date",
           y = "Feature Value",
           color = "Feature") +
      scale_color_manual(values = c("blue", "green", "red"))
  })
  
  # Generate heatmap for correlation
  output$heatmap <- renderPlot({
    # Store the selected data in variables
    data_X <- filtered_data_X()
    data_Y <- filtered_data_Y()
    data_Z <- filtered_data_Z()
    data_X <- subset(data_X, select = -date)
    data_Y <- subset(data_Y, select = -date)
    data_Z <- subset(data_Z, select = -date)
    data_X <- data_X[, input$feature_X, drop = FALSE]
    data_Y <- data_Y[, input$feature_Y, drop = FALSE]
    data_Z <- data_Z[, input$feature_Z, drop = FALSE]
    
    data_combined <- cbind(data_X, data_Y, data_Z)
    
    correlation_matrix <- cor(data_combined)
    
    ggplot(data = melt(correlation_matrix), aes(Var1, Var2, fill = value)) +
      geom_tile() +
      theme_minimal() +
      labs(title = "Correlation Heatmap",
           x = "Feature",
           y = "Feature",
           fill = "Correlation")
  })
}

# Run the application
shinyApp(ui, server)