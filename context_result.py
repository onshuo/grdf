import time
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.vector_stores.base import BaseVectorStore
from graphrag.utils.embeddings import create_collection_name
from graphrag.vector_stores.factory import VectorStoreFactory
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
import tiktoken
from graphrag.query.llm.get_client import get_llm, get_text_embedder
import pandas as pd
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.create_pipeline_config import create_pipeline_config
from graphrag.storage.factory import StorageFactory
from graphrag.utils.storage import load_table_from_storage
from pathlib import Path
from graphrag.config.load_config import load_config
from graphrag.config.resolve_path import resolve_paths
import json


# 全局变量，用于存储 context_builder
CONTEXT_BUILDER = None
GLOBAL_CONFIG = None
ROOT_PATH = "/aaa/bbb/ccc"


async def initialize_context_builder():
    """初始化 context_builder 并存储为全局变量"""
    global CONTEXT_BUILDER
    global GLOBAL_CONFIG

    root_dir = Path(ROOT_PATH)
    root_dir.resolve()
    GLOBAL_CONFIG = load_config(root_dir, None)
    resolve_paths(GLOBAL_CONFIG)

    dataframe_dict = await _resolve_output_files(
        config=GLOBAL_CONFIG,
        output_list=[
            "create_final_nodes.parquet",
            "create_final_community_reports.parquet",
            "create_final_text_units.parquet",
            "create_final_relationships.parquet",
            "create_final_entities.parquet",
        ],
    )

    final_community_reports = dataframe_dict["create_final_community_reports"]
    final_nodes = dataframe_dict["create_final_nodes"]
    final_relationships = dataframe_dict["create_final_relationships"]
    final_text_units = dataframe_dict["create_final_text_units"]
    final_entities = dataframe_dict["create_final_entities"]
    final_covariates = dataframe_dict.get("create_final_covariates", None)

    community_level = 2
    covariates_ = (
        read_indexer_covariates(final_covariates)
        if final_covariates is not None
        else []
    )
    entities_ = read_indexer_entities(final_nodes, final_entities, community_level)

    description_embedding_store = await _get_embedding_store(
        config_args=GLOBAL_CONFIG.embeddings.vector_store,
        embedding_name="entity.description",
    )
    text_embedder = get_text_embedder(GLOBAL_CONFIG)
    token_encoder = tiktoken.get_encoding(GLOBAL_CONFIG.encoding_model)

    CONTEXT_BUILDER = LocalSearchMixedContext(
        community_reports=read_indexer_reports(
            final_community_reports, final_nodes, community_level
        ),
        text_units=read_indexer_text_units(final_text_units),
        entities=entities_,
        relationships=read_indexer_relationships(final_relationships),
        covariates={"claims": covariates_},
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    print("Context builder initialized successfully.")


async def _resolve_output_files(
    config: GraphRagConfig,
    output_list: list[str],
    optional_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read indexing output files to a dataframe dict."""
    dataframe_dict = {}
    pipeline_config = create_pipeline_config(config)
    storage_config = pipeline_config.storage.model_dump()  # type: ignore
    storage_obj = StorageFactory().create_storage(
        storage_type=storage_config["type"], kwargs=storage_config
    )
    for output_file in output_list:
        df_key = output_file.split(".")[0]
        df_value = await load_table_from_storage(name=output_file, storage=storage_obj)
        dataframe_dict[df_key] = df_value

    # for optional output files, set the dict entry to None instead of erroring out if it does not exist
    if optional_list:
        for optional_file in optional_list:
            file_exists = await storage_obj.has(optional_file)
            df_key = optional_file.split(".")[0]
            if file_exists:
                df_value = await load_table_from_storage(
                    name=optional_file, storage=storage_obj
                )
                dataframe_dict[df_key] = df_value
            else:
                dataframe_dict[df_key] = None

    return dataframe_dict


async def _get_embedding_store(
    config_args: dict,
    embedding_name: str,
) -> BaseVectorStore:
    """Get the embedding description store."""
    vector_store_type = config_args["type"]
    collection_name = create_collection_name(
        config_args.get("container_name", "default"), embedding_name
    )
    embedding_store = VectorStoreFactory().create_vector_store(
        vector_store_type=vector_store_type,
        kwargs={**config_args, "collection_name": collection_name},
    )
    embedding_store.connect(**config_args)
    return embedding_store


# 自定义序列化逻辑
async def dataframe_dict_to_json(data):
    # 将 DataFrame 转换为字典
    converted = {key: df.to_dict(orient="records") for key, df in data.items()}
    # 转换为 JSON 字符串
    jsonstr = json.dumps(converted, indent=4, ensure_ascii=False)

    return jsonstr


# 自定义序列化逻辑
async def dataframe_dict_to_dict(data):
    # 将 DataFrame 转换为字典
    converted = {key: df.to_dict(orient="records") for key, df in data.items()}
    # 转换为 JSON 字符串
    jsonstr = json.dumps(converted, indent=4, ensure_ascii=False)

    return json.loads(jsonstr)


async def query_graphrag_context(query: str):
    # start_time = time.time()
    # search_prompt = ""

    # root_dir = Path("/Volumes/OnshuoData/graphrag")
    # root_dir.resolve()
    # config = load_config(root_dir, None)
    # resolve_paths(config)

    # dataframe_dict = await _resolve_output_files(
    #     config=config,
    #     output_list=[
    #         "create_final_nodes.parquet",
    #         "create_final_community_reports.parquet",
    #         "create_final_text_units.parquet",
    #         "create_final_relationships.parquet",
    #         "create_final_entities.parquet",
    #     ],
    # )
    # final_community_reports: pd.DataFrame = dataframe_dict[
    #     "create_final_community_reports"
    # ]
    # final_nodes: pd.DataFrame = dataframe_dict["create_final_nodes"]

    # final_community_reports: pd.DataFrame = dataframe_dict[
    #     "create_final_community_reports"
    # ]
    # final_relationships: pd.DataFrame = dataframe_dict["create_final_relationships"]
    # final_text_units: pd.DataFrame = dataframe_dict["create_final_text_units"]
    # final_entities: pd.DataFrame = dataframe_dict["create_final_entities"]
    # final_covariates: pd.DataFrame = dataframe_dict.get("create_final_covariates", None)

    # community_level=2
    # covariates_ = read_indexer_covariates(final_covariates) if final_covariates is not None else []
    # entities_ = read_indexer_entities(final_nodes, final_entities, community_level)
    # print(config.embeddings.vector_store)
    # description_embedding_store = await _get_embedding_store(
    #     config_args=config.embeddings.vector_store,  # type: ignore
    #     embedding_name="entity.description",
    # )
    # text_embedder = get_text_embedder(config)
    # token_encoder = tiktoken.get_encoding(config.encoding_model)
    # context_builder = LocalSearchMixedContext(
    #     community_reports=read_indexer_reports(final_community_reports, final_nodes,community_level),
    #     text_units=read_indexer_text_units(final_text_units),
    #     entities=entities_,
    #     relationships=read_indexer_relationships(final_relationships),
    #     covariates={"claims": covariates_},
    #     entity_text_embeddings=description_embedding_store,
    #     embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
    #     text_embedder=text_embedder,
    #     token_encoder=token_encoder,
    # )
    ls_config = GLOBAL_CONFIG.local_search

    context_result = CONTEXT_BUILDER.build_context(
        query=query,
        conversation_history=None,
        **{},
        **{
            "text_unit_prop": ls_config.text_unit_prop,
            "community_prop": ls_config.community_prop,
            "conversation_history_max_turns": ls_config.conversation_history_max_turns,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": ls_config.top_k_entities,
            "top_k_relationships": ls_config.top_k_relationships,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
            "max_tokens": ls_config.max_tokens,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        },
    )
    # return await dataframe_dict_to_dict(context_result.context_records)
    return {"chunks" : context_result.context_chunks}


def main():
    query_graphrag_context()


if __name__ == "__main__":
    main()
