from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class Singleton(type):
    """Singleton metaclass."""

    _instances: Dict[Type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)  # type: ignore
        return cls._instances[cls]


class ObjectFactory(Generic[T]):
    """Factory for creating objects."""

    def __init__(self, cls: T, *args, **kwargs):
        self.cls = cls
        self._base_args = args
        self._base_kwargs = kwargs

    def call_classmethod(self, method: str, *args, **kwargs) -> Any:
        """Call a classmethod on the class"""

        args, kwargs = self._compile_arguments(args, kwargs)
        return getattr(self.cls, method)(*args, **kwargs)

    def _compile_arguments(self, args: Tuple, kwargs: Dict) -> Tuple[List, Dict]:
        use_args = []

        for v in self._base_args:
            if callable(v):
                use_args.append(v())
            else:
                use_args.append(v)

        for k, v in self._base_kwargs.items():
            if k in kwargs:
                raise ValueError(f"Duplicate kwarg {k}={v} for {self.cls}")

            if callable(v):
                kwargs[k] = v()
            else:
                kwargs[k] = v

        return use_args + list(args), kwargs

    def __call__(self, *args, **kwargs) -> T:
        args, kwargs = self._compile_arguments(args, kwargs)
        return self.cls(*args, **kwargs)


class ObjectFactoryRegistry(Dict[Type, ObjectFactory], metaclass=Singleton):
    """Global Object factory registry"""

    @classmethod
    def add_factory(cls, factory: ObjectFactory, index_cls: Optional[Type] = None):
        self = cls()
        self[index_cls or factory.cls] = factory

    @classmethod
    def add(cls, fac_cls: Type, *args, index_cls: Optional[Type] = None, **kwargs):
        factory = ObjectFactory(fac_cls, *args, **kwargs)
        cls.add_factory(factory, index_cls=index_cls)

    @classmethod
    def fetch(cls, factory: Type) -> ObjectFactory:
        self = cls()
        return self[factory]
