import time


def call_threedi_api(func, max_retries: int = 60, **kwargs):
    """Add something to simulation, if apiexcetion is raised sleep on it and try again."""
    from threedi_api_client.openapi import ApiException  # Import here so its not a forced dependency

    for i in range(max_retries):
        try:
            # check if data is dict
            if "data" in kwargs.keys():
                if not isinstance(kwargs["data"], dict):
                    print("OJEE... data is geen dict {func}")
                    break

            r = func(**kwargs)
            return r
        except ApiException as e:
            error = e
            print(e)
            if error.status == 400:
                print(f"ERROR in {func}")
                print(error.body)
                raise e
            else:  # TODO add error code of API overload
                time.sleep(10)
                continue
