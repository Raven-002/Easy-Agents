#!/usr/bin/env python

import pytest

from easy_agents.agents.code.symbol_analyzer import SymbolAnalysisRequest, symbol_analyzer
from easy_agents.core import Router, ToolDepsRegistry
from easy_agents.tools.deps.project_files_deps import ProjectFilesDeps, project_files_deps_type


@pytest.mark.asyncio
async def test_agent(complex_router: Router, pytestconfig: pytest.Config) -> None:
    result = await symbol_analyzer.run(
        SymbolAnalysisRequest(symbol_name="Agent"),
        complex_router,
        deps=ToolDepsRegistry.from_map(
            {project_files_deps_type: ProjectFilesDeps(project_root=str(pytestconfig.rootpath))}
        ),
    )
    print(result)
