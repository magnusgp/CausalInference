# Read .csv file from the sample directory and load it into the database
def sampleLoad(filepath):
    # Import csv file and read it
    import csv
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Create a list of the rows in the file
        rows = list(reader)
        # Create a list of the headers in the file
        headers = rows[0]
        if headers[0] == '':
            print('Setting ID column')
            headers[0] = 'id'
        # Create a list of the data in the file
        data = rows[1:]
        # Print data summary
        print("\nData Summary:")
        print("\tNumber of rows:", len(data))
        print("\tNumber of columns:", len(headers))
        print("\tHeaders:", headers)
        return data

if __name__ == '__main__':
    # Get the filepath from the user
    path = 'sample/data_435.csv'
    # Call the function
    sampleLoad(path)


