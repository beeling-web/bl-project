import os

## This function will be used at main.py to save the both question and answer into the text file for future reference
## This is suggested by trainer to minimise calling to the AI model
def save_content(filepath: str, new_content: str):

    existing_content = ""
    file_exists = os.path.exists(filepath)

    # 1. Read existing content if the file exists
    if file_exists:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        except IOError:
            print(f"Error reading file {filepath}. Proceeding with overwrite.")

    # 2. Check for content difference 
    if existing_content == new_content:
        print(f"[*] File '{filepath}': Content is the same. No save required (Delta is zero).")
        return

    # 3. Write the new content (overwrite)
    try:
        # Use 'a' mode to create the file if it doesn't exist, or overwrite if it does.
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(new_content + "\n\n")
        
        # Report the action taken
        if file_exists:
            print(f"[!] File '{filepath}': Content changed. New content saved successfully.")
        else:
            print(f"[+] File '{filepath}': File created and content saved successfully.")

    except IOError as e:
        print(f"[X] An error occurred while writing to the file: {e}")


## This function will be used at main.py to retrieve the previous response from the text file if user ask the same question
def read_and_compare(filepath: str, comparison_content: str) -> str:
    
    comparison_content = comparison_content.strip()
    
    if not os.path.exists(filepath):
        print(f"[-] File '{filepath}' does not exist.")
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # Define the stopping boundary for the revious AI response
        end_marker = "Thanks for asking!"

        # Convert both content strings to lowercase for case-insensitive search
        lower_file_content = file_content.lower()
        lower_comparison_content = comparison_content.lower()


        # 1. Find the starting index of the comparison content
        # We search in the lowercase string but use the index on the original string.
        start_index = lower_file_content.find(lower_comparison_content)


        if start_index != -1:
            # Calculate the index right after the comparison content ends
            content_after_index = start_index + len(comparison_content)

            # In the text file, The AI reponse would start after a single quote (for closing up the question) \
            # and a new line, we do not need to display the single quote and new line to the user 
            
            # Check for the 3-character skip pattern: [1 char] + ['\n'] (Punctuation present)
            is_three_char_skip = (
                len(file_content) >= content_after_index + 3 and
                file_content[content_after_index + 1] == "'" and
                file_content[content_after_index + 2] == '\n'
            )
            
            # Check for the 2-character skip pattern: ['\n'] (Punctuation absent)
            is_two_char_skip = (
                len(file_content) >= content_after_index + 2 and
                file_content[content_after_index] == "'" and
                file_content[content_after_index + 1] == '\n'
            )

            if is_three_char_skip:
                print("[*] Detected the 'one char + quote + newline' pattern (X'\\n). Advancing start position by 3.")
                # Skip the character (X), the quote ('), and the newline (\n)
                content_after_index += 3 
            elif is_two_char_skip:
                print("[*] Detected the 'quote + newline' pattern ('\\n). Advancing start position by 2.")
                # Skip the quote (') and the newline (\n)
                content_after_index += 2


            # 2. Get the remaining content after the start key
            content_from_start = file_content[content_after_index:]
            
            # 3. Find the index of the end marker within the remaining content
            end_index = content_from_start.find(end_marker)
            
            if end_index != -1:
                # End marker found: Extract content up to the marker
                extracted_content = content_from_start[:end_index + len(end_marker)]
                print(f"[*] Found comparison string and end marker. Extracted content stopping before '{end_marker}'.")
            else:
                # End marker not found: Return the rest of the file
                extracted_content = content_from_start
                print(f"[*] Found comparison string, but end marker was not found. Extracted content until EOF.")
            
            return extracted_content
        else:
            print(f"[-] Comparison string was NOT found in the file content.")
            return ""

    except IOError as e:
        print(f"[X] An error occurred while reading the file: {e}")
        return ""
