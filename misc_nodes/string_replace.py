"""
Gen2 String Utility Nodes
"""


class Gen2_StringReplace:
    """
    Replace all occurrences of a search string with a replacement string.
    
    Case-sensitive exact match replacement.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True}),
                "search": ("STRING", {"default": ""}),
                "replace": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_string",)
    FUNCTION = "replace_string"
    CATEGORY = "Gen2/Utils"
    
    def replace_string(self, input_string: str, search: str, replace: str) -> tuple:
        """
        Replace all occurrences of search string with replace string.
        
        Args:
            input_string: The input string to process
            search: The exact string to search for (case-sensitive)
            replace: The string to replace matches with
        
        Returns:
            Tuple containing the processed string
        """
        if not search:
            # If search is empty, return input unchanged
            return (input_string,)
        
        output = input_string.replace(search, replace)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Gen2_StringReplace": Gen2_StringReplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_StringReplace": "Gen2 StringReplace",
}

