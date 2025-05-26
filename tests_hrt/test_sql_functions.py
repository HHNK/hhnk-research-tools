# %%
from hhnk_research_tools.sql_functions import sql_builder_select_by_id_list_statement


def test_sql_builder_select_by_id_list_statement():
    sub_id_list_sql = """SELECT SUB_ID FROM SCHEMA.SUPERTABLE"""
    schema = "SCHEMA"
    sub_table = "SUBTABLE"
    sub_id_column = "ID"

    sql = sql_builder_select_by_id_list_statement(
        sub_id_list_sql=sub_id_list_sql,
        schema=schema,
        sub_table=sub_table,
        sub_id_column=sub_id_column,
    )
    assert sql == "SELECT *\nFROM SCHEMA.SUBTABLE\nWHERE ID IN (\n    SELECT SUB_ID FROM SCHEMA.SUPERTABLE \n)"


if __name__ == "__main__":
    test_sql_builder_select_by_id_list_statement()
