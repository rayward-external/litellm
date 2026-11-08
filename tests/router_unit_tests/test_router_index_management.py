import os
import pytest
import ast

from litellm import Router


class TestRouterIndexManagement:
    """Test cases for router index management functions"""

    def test_no_linear_scans_in_router(self):
        """
        Static analysis test to ensure Router doesn't use O(n) linear scans.

        Scans router.py for 'in self.model_list' pattern which indicates
        inefficient O(n) iteration instead of using index-based O(1) lookups.

        Methods should use:
        - model_id_to_deployment_index_map for O(1) model_id lookups
        - model_name_to_deployment_indices for O(1) + O(k) model_name lookups
        """
        # Methods that are allowed to iterate through self.model_list
        ALLOWED_METHODS = {
            "_get_deployment_by_litellm_model": "lookup by litellm_params.model, which is not indexed",
            "_finalize_adaptive_router_if_configured": 'init-time prefix scan for "auto_router/adaptive_router"; no index for prefix match',
            "config_deployments": "filters the whole list on model_info.db_model; admin path only (model add/upsert)",
            "auto_router_capability_violation": "counts gated auto-routers across the whole list; admin path only (auto-router init/upsert)",
        }

        # Get path to router.py
        router_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "litellm",
            "router.py",
        )

        # Read the file
        with open(router_file, "r") as f:
            content = f.read()

        # Parse with AST
        tree = ast.parse(content)

        # Find violations
        violations = []
        ignore_methods = set(ALLOWED_METHODS)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_name = node.name

                # Skip ignored methods
                if method_name in ignore_methods:
                    continue

                # Get source for this method
                try:
                    method_source = ast.get_source_segment(content, node)
                    if not method_source:
                        continue

                    # Check for the anti-pattern: "in self.model_list"
                    # This catches: for x in self.model_list, if x in self.model_list, etc.
                    if "in self.model_list" in method_source:
                        # Extract the specific line for better error reporting
                        lines = method_source.split("\n")
                        pattern_line = None
                        for line in lines:
                            if "in self.model_list" in line:
                                pattern_line = line.strip()
                                break

                        violations.append(
                            {
                                "method": method_name,
                                "line": node.lineno,
                                "pattern": pattern_line or "in self.model_list",
                            }
                        )
                except Exception:
                    # Skip if we can't get source segment
                    pass

        # Assert no violations
        if violations:
            error_msg = "\n".join(
                [
                    f"  - {v['method']}() at line {v['line']}: {v['pattern']}"
                    for v in violations
                ]
            )

            pytest.fail(
                f"\n{'='*70}\n"
                f"Found O(n) linear scan pattern in router.py:\n\n"
                f"{error_msg}\n\n"
                f"These methods should use index maps instead:\n"
                f"  - model_id_to_deployment_index_map (for model_id lookups)\n"
                f"  - model_name_to_deployment_indices (for model_name lookups)\n\n"
                f"If a method legitimately needs O(n) iteration, add it to\n"
                f"ALLOWED_METHODS in this test method.\n"
                f"{'='*70}\n"
            )

    def test_model_names_is_set(self):
        """Verify that model_names uses a set for O(1) lookups, not a list (O(n))"""
        router = Router(model_list=[])

        assert isinstance(
            router.model_names, set
        ), f"model_names should be a set for O(1) lookups, but got {type(router.model_names)}"
