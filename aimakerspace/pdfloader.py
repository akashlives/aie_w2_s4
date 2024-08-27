from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def __init__(self, file_path: str):
        """
        Initialize the PDFLoader class with the path to the PDF file.

        Args:
            file_path (str): The path to the PDF file to be loaded.
        """
        self.file_path = file_path
        self.loader = PyPDFLoader(self.file_path)
        self.pages = None

    def load_and_split(self):
        """
        Load and split the PDF file into pages.

        Returns:
            list: A list of pages after loading and splitting the PDF file.
        """
        self.pages = self.loader.load_and_split()
        return self.pages

    def get_page(self, page_number: int):
        """
        Get a specific page from the loaded PDF.

        Args:
            page_number (int): The page number to retrieve.

        Returns:
            dict: The content of the specified page.
        """
        if self.pages is None:
            raise ValueError(
                "The PDF has not been loaded yet. Call load_and_split() first."
            )

        if page_number < 1 or page_number > len(self.pages):
            raise ValueError(
                f"Page number out of range. Please choose a value between 1 and {len(self.pages)}."
            )

        return self.pages[page_number - 1]

    def get_total_pages(self):
        """
        Get the total number of pages in the PDF.

        Returns:
            int: The total number of pages.
        """
        if self.pages is None:
            raise ValueError(
                "The PDF has not been loaded yet. Call load_and_split() first."
            )

        return len(self.pages)
