# Personal-Finance-Tracker


# Personal Finance Tracker

## Description
The Personal Finance Tracker is a web application that helps users manage their finances by tracking expenses, income, and setting financial goals. It provides an easy way to categorize transactions, generate reports, and visualize spending patterns, enabling users to make informed financial decisions.

## Features
- Track income and expenses with categorized entries.
- Visualize spending patterns with charts and graphs.
- Set and monitor financial goals (e.g., saving targets).
- View monthly and yearly summaries of financial activity.
- Secure user authentication to protect financial data.

## Technologies Used
- **Python**: Backend programming language.
- **Flask**: Web framework for building the application.
- **SQLAlchemy**: ORM used for managing the database (SQLite or MySQL).
- **HTML/CSS/Bootstrap**: Frontend design and styling.
- **Jinja2**: For dynamic templating.
- **Matplotlib/Plotly**: For creating visualizations of the financial data.

## Setup and Installation

### Prerequisites
- Python 3.x installed on your system.
- A virtual environment (recommended but optional).
- A database setup (default: SQLite, or configure for MySQL).

### How to Run Locally
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/personal-finance-tracker.git
    cd personal-finance-tracker
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure the database**:
    - By default, the app is configured to use SQLite. If you wish to use MySQL, update the configuration in the `config.py` file with your database credentials.
    
5. **Run the database migrations**:
    ```bash
    flask db upgrade
    ```

6. **Run the application**:
    ```bash
    python app.py
    ```

7. **Access the application**:
    Open a web browser and go to:
    ```
    http://127.0.0.1:5000
    ```


## Future Improvements
- Add support for multiple currencies and exchange rates.
- Implement budget notifications and reminders.
- Integrate with external banking APIs to automatically sync transactions.
- Allow users to export financial reports in PDF or CSV format.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


