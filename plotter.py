import psycopg2
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import allantools

# Database connection parameters - Update these with your credentials
DB_HOST = "192.168.120.100"
DB_NAME = "spacedb"
DB_USER = "postgres"
DB_PASSWORD = ""

# Query parameters - Update these with your desired values
PAIR_ID = 1  # Replace with your pair_id
START_TIME = datetime(2025, 9, 19, 17, 0, 0)  # Start of time range
END_TIME = datetime(2025, 9, 23, 20, 0, 0)  # End of time range


def detrend(x, n=1):
    x_detrend = x.copy()
    try:
        t = np.arange(len(x_detrend))*100
        p = np.polyfit(t, x_detrend, n)        
        print("p = ", p)
        print("x_end = ", x[-1])
        x_detrend -= np.polyval(p, t)
    except:
        print("detrend error")
    return x_detrend   


def fetch_phase_data():
    """Fetch phase data from the database"""
    conn = None
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Create a cursor
        cur = conn.cursor()
        
        # SQL query to fetch phase data
        query = """
            SELECT timestamp, phase 
            FROM raw_phase 
            WHERE pair_id = %s 
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
        """
        
        # Execute the query
        cur.execute(query, (PAIR_ID, START_TIME, END_TIME))
        
        # Fetch all results
        data = cur.fetchall()
        
        # Close communication with the database
        cur.close()
        
        return data
    
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database error occurred: {error}")
        return None
    
    finally:
        if conn is not None:
            conn.close()


def plot_phase_data(data):
    """Plot phase data using matplotlib"""
    if not data:
        print("No data to plot")
        return
    
    # Unpack data into separate lists
    #tics = np.array([row[0].timestamp() for row in data])
    timestamps = [row[0] for row in data]
    phases = np.array([row[1] for row in data])
    
    print(timestamps[-1])
    
    
    # Create plot
    plt.figure(figsize=(12, 6))
    #plt.plot(tics[1:]-tics[:-1], marker='o', linestyle='-', markersize=4)
    phases = np.where(phases > 0.5, phases - 1., phases)
    plt.plot(timestamps, (phases), marker='o', linestyle='-', markersize=4)        
        
    # Format plot
    plt.title(f'Phase Data for Pair ID {PAIR_ID}')
    plt.xlabel('Timestamp')
    plt.ylabel('Phase (s)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show plot
    #plt.show(block=False)    
    
    plt.figure(figsize=(12, 6))
    phases = np.array(phases)    
    kdec = 10
    dphases = phases[kdec::1] - phases[:-kdec:1]
    #dtimes = timestamps[10::1] - timestamps[:-10:1]
    y = dphases/100/kdec
    n = len(dphases)
    plt.plot(timestamps[:n], y, marker='o', linestyle='-', markersize=4)        
        
    # Format plot
    plt.title(f'Freq Data for Pair ID {PAIR_ID}')
    plt.xlabel('Timestamp')
    plt.ylabel('Freq (s)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show plot
    #plt.show(block=False)    


def calculate_allan_deviation(data):
    """Calculate Allan deviation using allantools"""
    if not data or len(data) < 2:
        print("Insufficient data for Allan deviation calculation")
        return None, None
    
    # Extract phase values and timestamps
    phases = np.array([row[1] for row in data])
    timestamps = np.array([row[0] for row in data])
    
    # Calculate average sampling period
    time_diffs = np.diff(timestamps).astype('timedelta64[s]').astype(float)
    tau0 = np.mean(time_diffs)  # Average time between samples in seconds
    
    if tau0 <= 0:
        print("Invalid sampling interval detected")
        return None, None
    
    # Calculate Allan deviation
    rate = 1/tau0  # Sampling rate in Hz
    (taus, adevs, _, _) = allantools.adev(
        phases,
        rate=rate,
        data_type="phase",
        taus="decade"  # Automatically choose tau values
    )
    
    return taus, adevs

def plot_allan_deviation(taus, adevs):
    """Plot Allan deviation results"""
    plt.figure(figsize=(12, 6))
    plt.loglog(taus, adevs, 'o-')
    plt.title(f'Allan Deviation for Pair ID {PAIR_ID}')
    plt.xlabel('Averaging Time (s)')
    plt.ylabel('Allan Deviation')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    #plt.show(block=False)


if __name__ == "__main__":
    # Fetch data from database
    phase_data = fetch_phase_data()
    
    # Plot the data if available
    if phase_data:
        plot_phase_data(phase_data)
        
        taus, adevs = calculate_allan_deviation(phase_data)
        if taus is not None and adevs is not None:
            plot_allan_deviation(taus, adevs)
        
        plt.show()
    else:
        print("No data found for the specified parameters")