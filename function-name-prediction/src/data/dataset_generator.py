import pandas as pd
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _count_params(params: str) -> int:
    if not params or not params.strip():
        return 0
    return len([part for part in params.split(",") if part.strip()])

def _build_description_variants(func: dict, base_description: str, params: str) -> list:
    keyword_hint = ", ".join(func["keywords"][:2])
    return [
        base_description,
        f"{base_description}. Uses parameters {params}.",
        f"{base_description}. Returns {func['return_type']}.",
        f"{base_description}. Library {func['library']} operation.",
        f"{base_description}. Related keywords: {keyword_hint}.",
    ]

# Define the structure and metadata for dataset generation
FUNCTION_TEMPLATES = {
    "MathUtils": [
        {
            "function_name": "addNumbers",
            "descriptions": [
                "Adds two integers and returns their sum",
                "Calculates the sum of two integers",
                "Performs addition on two numeric inputs",
                "Returns the total when adding two numbers together",
                "Computes the addition of a and b"
            ],
            "parameters": ["int a, int b", "int num1, int num2", "integer x, integer y", "int first, int second"],
            "return_type": "int",
            "keywords": ["add", "sum", "arithmetic", "math", "addition"]
        },
        {
            "function_name": "subtractNumbers",
            "descriptions": [
                "Subtracts the second number from the first",
                "Calculates the difference between two values",
                "Computes the subtraction of b from a",
                "Returns the result of subtracting two integers",
                "Finds the difference between num1 and num2"
            ],
            "parameters": ["int a, int b", "int num1, int num2", "int minuend, int subtrahend"],
            "return_type": "int",
            "keywords": ["subtract", "difference", "minus", "math"]
        },
        {
            "function_name": "multiplyNumbers",
            "descriptions": [
                "Multiplies two numbers together",
                "Returns the product of the given values",
                "Computes the multiplication of two integers",
                "Finds the product of a and b",
                "Calculates x times y"
            ],
            "parameters": ["int a, int b", "int num1, int num2", "float x, float y"],
            "return_type": "int",  # or float in some variations, we'll keep it static here
            "keywords": ["multiply", "product", "times", "math"]
        },
        {
            "function_name": "divideNumbers",
            "descriptions": [
                "Divides the first number by the second",
                "Calculates the quotient of two values",
                "Computes the division of numerator by denominator",
                "Returns a divided by b",
                "Performs a division operation"
            ],
            "parameters": ["float a, float b", "double numerator, double denominator", "float num1, float num2"],
            "return_type": "float",
            "keywords": ["divide", "quotient", "math", "fraction"]
        },
        {
            "function_name": "calculateAverage",
            "descriptions": [
                "Calculates the average of a list of numbers",
                "Finds the mean value from an array",
                "Returns the arithmetic mean of given items",
                "Computes the average score",
                "Determines the average of the elements"
            ],
            "parameters": ["List<float> numbers", "float[] values", "List<int> items"],
            "return_type": "float",
            "keywords": ["average", "mean", "math", "statistics"]
        },
        {
            "function_name": "findMaximum",
            "descriptions": [
                "Finds the maximum value in a collection",
                "Returns the largest number in an array",
                "Retrieves the max value from the given list",
                "Identifies the highest number",
                "Gets the maximum integer"
            ],
            "parameters": ["List<int> numbers", "int[] array", "List<float> items"],
            "return_type": "int",
            "keywords": ["max", "maximum", "largest", "highest", "math"]
        },
        {
            "function_name": "findMinimum",
            "descriptions": [
                "Finds the minimum value in a collection",
                "Returns the smallest number in an array",
                "Retrieves the min value from the given list",
                "Identifies the lowest number",
                "Gets the smallest integer"
            ],
            "parameters": ["List<int> numbers", "int[] array", "List<float> items"],
            "return_type": "int",
            "keywords": ["min", "minimum", "smallest", "lowest", "math"]
        }
    ],
    "StringUtils": [
        {
            "function_name": "reverseString",
            "descriptions": [
                "Reverses a given string",
                "Returns the characters of a string in reverse order",
                "Inverts a text sequence",
                "Creates a reversed copy of the string",
                "Flips the characters in the string"
            ],
            "parameters": ["String text", "String input", "String str", "char[] chars"],
            "return_type": "String",
            "keywords": ["reverse", "string", "invert", "text"]
        },
        {
            "function_name": "capitalizeString",
            "descriptions": [
                "Capitalizes the first letter of a string",
                "Converts the string to title case",
                "Makes the first character of the text uppercase",
                "Ensures the string starts with a capital letter",
                "Transforms string to capitalized case"
            ],
            "parameters": ["String text", "String inputText", "String str"],
            "return_type": "String",
            "keywords": ["capitalize", "uppercase", "string", "format"]
        },
        {
            "function_name": "splitString",
            "descriptions": [
                "Splits a string by a given delimiter",
                "Separates text into an array based on a separator",
                "Divides a string into multiple parts",
                "Breaks down a string into a list",
                "Tokensizes the string using delimiter"
            ],
            "parameters": ["String text, String delimiter", "String input, char separator", "String str, String regex"],
            "return_type": "List<String>",
            "keywords": ["split", "separate", "string", "delimiter"]
        },
        {
            "function_name": "joinStrings",
            "descriptions": [
                "Joins an array of strings into one",
                "Concatenates elements of a list using a delimiter",
                "Merges a list of strings into a single text",
                "Combines multiple strings with a separator",
                "Reconstructs a string from parts"
            ],
            "parameters": ["List<String> parts, String separator", "String[] items, String delimiter", "List<String> list, String joiner"],
            "return_type": "String",
            "keywords": ["join", "string", "concat", "merge"]
        },
        {
            "function_name": "trimWhitespace",
            "descriptions": [
                "Removes leading and trailing spaces from a string",
                "Trims whitespace from both ends of text",
                "Cleans up extra spaces at the beginning and end",
                "Strips whitespace characters",
                "Returns string without leading or trailing spaces"
            ],
            "parameters": ["String text", "String input", "String rawString"],
            "return_type": "String",
            "keywords": ["trim", "whitespace", "format", "clean"]
        },
        {
            "function_name": "countCharacters",
            "descriptions": [
                "Counts the number of characters in a string",
                "Returns the length of the provided text",
                "Gets the total character count",
                "Finds how many characters are in a string",
                "Calculates string length"
            ],
            "parameters": ["String text", "String input"],
            "return_type": "int",
            "keywords": ["count", "characters", "length", "string"]
        }
    ],
    "FileUtils": [
        {
            "function_name": "readFile",
            "descriptions": [
                "Reads all content from a file",
                "Loads text representation of a file",
                "Gets the file contents as a string",
                "Retrieves data from the specified path",
                "Parses file into a string format"
            ],
            "parameters": ["String filePath", "String path", "File file"],
            "return_type": "String",
            "keywords": ["read", "file", "content", "io"]
        },
        {
            "function_name": "writeFile",
            "descriptions": [
                "Writes text data to a file on disk",
                "Saves content into a file",
                "Outputs the given string to a file path",
                "Stores string data in the filesystem",
                "Generates a file with provided content"
            ],
            "parameters": ["String filePath, String content", "String path, String data", "File file, String text"],
            "return_type": "void",
            "keywords": ["write", "save", "file", "io", "output"]
        },
        {
            "function_name": "deleteFile",
            "descriptions": [
                "Deletes a file from the filesystem",
                "Removes a file given its path",
                "Trashes the specified file",
                "Clears a file from the disk",
                "Erases the file completely"
            ],
            "parameters": ["String filePath", "String targetPath", "File file"],
            "return_type": "boolean",
            "keywords": ["delete", "remove", "file", "erase"]
        },
        {
            "function_name": "copyFile",
            "descriptions": [
                "Copies a file from source to destination",
                "Duplicates an existing file",
                "Creates a copy of the file at the new path",
                "Replicates file data to another location",
                "Transfers file copy across directories"
            ],
            "parameters": ["String source, String destination", "String srcPath, String destPath", "File sourceFile, File targetFile"],
            "return_type": "boolean",
            "keywords": ["copy", "duplicate", "file", "transfer"]
        },
        {
            "function_name": "moveFile",
            "descriptions": [
                "Moves a file to a new location",
                "Relocates a file on the disk",
                "Renames or moves a file path",
                "Transfers file permanently to another directory",
                "Shifts file to a destination path"
            ],
            "parameters": ["String source, String destination", "String oldPath, String newPath"],
            "return_type": "boolean",
            "keywords": ["move", "relocate", "file", "rename"]
        },
        {
            "function_name": "fileExists",
            "descriptions": [
                "Checks if a file exists at the given path",
                "Returns true if the file is present on disk",
                "Validates whether a generic file path is accessible",
                "Determines if a file exists",
                "Verifies the file's existence"
            ],
            "parameters": ["String filePath", "String path", "String location"],
            "return_type": "boolean",
            "keywords": ["file", "exists", "check", "verify"]
        }
    ],
    "NetworkService": [
        {
            "function_name": "sendGetRequest",
            "descriptions": [
                "Performs an HTTP GET request",
                "Fetches data from a URL using GET",
                "Sends a GET request to the specified endpoint",
                "Retrieves an HTTP response securely",
                "Executes a network GET call"
            ],
            "parameters": ["String url", "String endpoint", "URI uri"],
            "return_type": "HttpResponse",
            "keywords": ["get", "http", "request", "network"]
        },
        {
            "function_name": "sendPostRequest",
            "descriptions": [
                "Sends an HTTP POST request with a payload",
                "Posts data to an API endpoint",
                "Executes a network POST operation",
                "Submits form or JSON data to a server",
                "Performs an HTTP POST"
            ],
            "parameters": ["String url, Object body", "String endpoint, String jsonPayload", "String uri, Map data"],
            "return_type": "HttpResponse",
            "keywords": ["post", "request", "http", "payload", "network"]
        },
        {
            "function_name": "fetchApiData",
            "descriptions": [
                "Fetches JSON data from an API",
                "Calls a REST endpoint to retrieve data",
                "Gets data from an external web API",
                "Retrieves structured data from the server",
                "Queries the API for results"
            ],
            "parameters": ["String endpoint", "String url", "String apiUrl"],
            "return_type": "JsonObject",
            "keywords": ["fetch", "api", "data", "json", "rest"]
        },
        {
            "function_name": "downloadFile",
            "descriptions": [
                "Downloads a file from the network",
                "Saves a remote file locally",
                "Fetches file bytes from a URL to local storage",
                "Retrieves a document from the internet",
                "Downloads media from a server"
            ],
            "parameters": ["String url, String destination", "String remoteUrl, String localPath"],
            "return_type": "boolean",
            "keywords": ["download", "file", "network", "fetch"]
        },
        {
            "function_name": "uploadFile",
            "descriptions": [
                "Uploads a local file to a server",
                "Sends file byte data via network",
                "Posts a document to an API endpoint",
                "Transmits local file to cloud storage",
                "Uploads media content"
            ],
            "parameters": ["String filePath, String url", "File file, String endpoint", "String localPath, String serverUrl"],
            "return_type": "boolean",
            "keywords": ["upload", "file", "network", "transfer"]
        }
    ],
    "DataUtils": [
        {
            "function_name": "sortList",
            "descriptions": [
                "Sorts a list of objects based on natural ordering",
                "Orders the elements in an array sequentially",
                "Arranges the list elements",
                "Returns a sorted copy of the collection",
                "Sorts given items in ascending order"
            ],
            "parameters": ["List<Object> items", "List<T> collection", "int[] array"],
            "return_type": "List<Object>",
            "keywords": ["sort", "order", "list", "arrange"]
        },
        {
            "function_name": "filterList",
            "descriptions": [
                "Filters a list based on a given condition",
                "Removes elements that do not match the predicate",
                "Keeps items matching criteria in the array",
                "Extracs matching elements",
                "Cleans up list via a filter function"
            ],
            "parameters": ["List<Object> items, Predicate condition", "List<T> collection, Function filterFunc"],
            "return_type": "List<Object>",
            "keywords": ["filter", "list", "remove", "select"]
        },
        {
            "function_name": "removeDuplicates",
            "descriptions": [
                "Removes duplicate values from a list",
                "Returns unique elements only from an array",
                "Strips out duplicated entries",
                "Purges repetitive items in the collection",
                "Deduplicates a list"
            ],
            "parameters": ["List<Object> items", "List<T> elements"],
            "return_type": "List<Object>",
            "keywords": ["unique", "duplicates", "clean", "list"]
        },
        {
            "function_name": "mergeLists",
            "descriptions": [
                "Merges two distinct lists into one",
                "Combines multiple arrays together",
                "Joins two collections sequentially",
                "Concatenates array sets into a single list",
                "Appends one list to another"
            ],
            "parameters": ["List<Object> list1, List<Object> list2", "List<T> a, List<T> b", "int[] first, int[] second"],
            "return_type": "List<Object>",
            "keywords": ["merge", "combine", "list", "join"]
        },
        {
            "function_name": "calculateSum",
            "descriptions": [
                "Calculates the total sum of items in a list",
                "Computes the overall total of numeric values",
                "Adds all elements in the array",
                "Totals the list values incrementally",
                "Returns the aggregate sum"
            ],
            "parameters": ["List<int> items", "List<float> numbers", "double[] array"],
            "return_type": "int",
            "keywords": ["sum", "total", "calculate", "list", "math"]
        }
    ],
    "ValidationUtils": [
        {
            "function_name": "validateEmail",
            "descriptions": [
                "Validates the format of an email address",
                "Checks if an email string is structurally sound",
                "Verifies whether an email contains valid characters and domain",
                "Determines if an address is properly formed",
                "Validates email utilizing regex"
            ],
            "parameters": ["String email", "String emailAddress", "String targetEmail"],
            "return_type": "boolean",
            "keywords": ["validate", "email", "check", "format"]
        },
        {
            "function_name": "validatePhone",
            "descriptions": [
                "Checks if the phone number is valid",
                "Verifies mobile number format",
                "Validates if the provided phone string is allowable",
                "Ensures the telephone follows proper length",
                "Confirms telephone numbers digit constraints"
            ],
            "parameters": ["String phoneNumber", "String mobile"],
            "return_type": "boolean",
            "keywords": ["validate", "phone", "number", "check"]
        },
        {
            "function_name": "validatePassword",
            "descriptions": [
                "Validates password strength according to rules",
                "Checks if the password meets security constraints",
                "Ensures password has appropriate complexity",
                "Verifies that string is a strong password",
                "Validates credentials threshold"
            ],
            "parameters": ["String password", "String rawPassword"],
            "return_type": "boolean",
            "keywords": ["validate", "password", "security", "check"]
        },
        {
            "function_name": "validateURL",
            "descriptions": [
                "Checks if a given string is a valid URL",
                "Verifies web address format",
                "Validates user-provided link structures",
                "Ensures the URL matches URI conventions",
                "Confirms the web address formatting"
            ],
            "parameters": ["String url", "String webAddress", "String link"],
            "return_type": "boolean",
            "keywords": ["validate", "url", "link", "web", "format"]
        },
        {
            "function_name": "isNumeric",
            "descriptions": [
                "Checks if a string is composed entirely of numbers",
                "Validates whether the text input is a numeric type",
                "Returns true if the string is parseable to number",
                "Tests digit membership inside a string",
                "Verifies numeric text representation"
            ],
            "parameters": ["String text", "String input", "String val", "String str"],
            "return_type": "boolean",
            "keywords": ["numeric", "number", "check", "validate", "digits"]
        }
    ],
    "TemperatureUtils": [
        {
            "function_name": "convertCelsiusToFahrenheit",
            "descriptions": [
                "Converts temperature from Celsius to Fahrenheit",
                "Convert Celsius temperature to Fahrenheit",
                "Transforms Celsius value into Fahrenheit",
                "Temperature conversion from C to F",
                "Converts degree Celsius reading to degree Fahrenheit",
                "Outputs Fahrenheit value for a Celsius input",
                "Maps Celsius measurement to Fahrenheit scale",
                "Calculate Fahrenheit from Celsius value"
            ],
            "parameters": ["float celsius", "double celsiusValue", "float tempC"],
            "return_type": "float",
            "keywords": ["temperature", "celsius", "fahrenheit", "convert", "c_to_f", "tofahrenheit", "celsius_to_fahrenheit"]
        },
        {
            "function_name": "convertFahrenheitToCelsius",
            "descriptions": [
                "Converts temperature from Fahrenheit to Celsius",
                "Convert Fahrenheit temperature to Celsius",
                "Transforms Fahrenheit value into Celsius",
                "Temperature conversion from F to C",
                "Converts degree Fahrenheit reading to degree Celsius",
                "Outputs Celsius value for a Fahrenheit input",
                "Maps Fahrenheit measurement to Celsius scale",
                "Calculate Celsius from Fahrenheit value"
            ],
            "parameters": ["float fahrenheit", "double fahrenheitValue", "float tempF"],
            "return_type": "float",
            "keywords": ["temperature", "fahrenheit", "celsius", "convert", "f_to_c", "tocelsius", "fahrenheit_to_celsius"]
        }
    ]
}

def generate_dataset(num_records=720):
    data = []
    
    # Flatten functions for easy looping
    flat_functions = []
    for library, funcs in FUNCTION_TEMPLATES.items():
        for func in funcs:
            func['library'] = library
            flat_functions.append(func)
            
    # Calculate how many variations per function to hit the target
    repeats_per_function = max(1, num_records // len(flat_functions))
    remainder = num_records % len(flat_functions)

    records_count = [repeats_per_function] * len(flat_functions)
    for i in range(remainder):
        records_count[i] += 1
        
    for idx, func in enumerate(flat_functions):
        library = func['library']
        func_name = func['function_name']
        ret_type = func['return_type']
        
        for _ in range(records_count[idx]):
            params = random.choice(func['parameters'])
            desc = random.choice(_build_description_variants(func, random.choice(func['descriptions']), params))
            kw = ", ".join(random.sample(func['keywords'], k=min(3, len(func['keywords']))))
            
            # calculate param count
            p_count = _count_params(params)
                
            row = {
                "description": desc,
                "parameters": params,
                "return_type": ret_type,
                "library": library,
                "keywords": kw,
                "param_count": p_count,
                "function_name": func_name
            }
            data.append(row)
            
    # Shuffle the dataset
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    return df

def main():
    print("Generating function names dataset...")
    df = generate_dataset(num_records=720)
    print(f"Generated {len(df)} records.")
    
    # Save the dataframe
    output_dir = PROJECT_ROOT / "data" / "raw"
    output_path = output_dir / "functions_dataset.csv"
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
