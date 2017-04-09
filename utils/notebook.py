def in_notebook():
    """Checks if we're currently running inside of an iPython Notebook.

    Based on http://stackoverflow.com/a/39662359/1828613.

    Returns
    -------
    in_notebook : Boolean
        True if the current code is executing within an iPython Notebook, or False otherwise.
    """
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        return False